/**
 * Live Audio Frequency Analysis API
 * 
 * Captures microphone audio in real-time, accumulates 1-second chunks,
 * and computes dominant frequency using FFT in a Web Worker.
 * 
 * Usage:
 *   const controller = await startLiveAudioAnalysis(({ peakFreq }) => {
 *     console.log(`Dominant frequency: ${peakFreq.toFixed(1)} Hz`);
 *   });
 *   // Later: await controller.stop();
 */

/**
 * Result passed to callback every second
 */
export interface LiveFrequencyResult {
  peakFreq: number;     // Dominant frequency in Hz
  peakVal: number;      // Magnitude at peak frequency
  timestamp: number;    // When this result was computed (ms since epoch)
}

/**
 * Controller for stopping live analysis
 */
export interface LiveAudioController {
  stop: () => Promise<void>;
}

/**
 * Callback function type for receiving frequency updates
 */
type FrequencyCallback = (result: LiveFrequencyResult) => void;

/**
 * Start live audio frequency analysis
 * 
 * Workflow:
 * 1. Request microphone access via getUserMedia
 * 2. Create AudioContext and load AudioWorklet processor
 * 3. Accumulate PCM samples until 1 second collected
 * 4. Transfer 1-second buffer to Web Worker for FFT analysis
 * 5. Worker computes dominant frequency and sends back result
 * 6. Invoke callback with result, repeat
 * 
 * @param onOneSecondResult - Callback invoked every ~1 second with frequency data
 * @returns Promise resolving to controller with stop() method
 */
export async function startLiveAudioAnalysis(
  onOneSecondResult: FrequencyCallback
): Promise<LiveAudioController> {
  // Request microphone access
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: false,  // We want raw audio
      noiseSuppression: false,
      autoGainControl: false
    }
  });

  // Create audio context
  const audioContext = new AudioContext();
  const sampleRate = audioContext.sampleRate;

  // Calculate how many samples = 1 second
  const samplesPerSecond = Math.round(sampleRate * 1.0);
  
  // Accumulation buffer for 1-second chunks
  let accumulatedSamples: Float32Array[] = [];
  let totalSamples = 0;

  // Create Web Worker for FFT computation
  const workerBlob = new Blob(
    [
      `
      // Inline worker code
      ${computeDominantFrequencyWorkerCode()}
      `
    ],
    { type: 'application/javascript' }
  );
  const workerUrl = URL.createObjectURL(workerBlob);
  const worker = new Worker(workerUrl);

  // Handle worker messages
  worker.onmessage = (event) => {
    if (event.data.type === 'result') {
      const { peakFreq, peakVal, timestamp } = event.data;
      onOneSecondResult({ peakFreq, peakVal, timestamp });
    }
  };

  // Process accumulated samples
  const processAccumulatedAudio = () => {
    if (totalSamples >= samplesPerSecond) {
      // Concatenate all accumulated chunks into single array
      const fullBuffer = new Float32Array(totalSamples);
      let offset = 0;
      for (const chunk of accumulatedSamples) {
        fullBuffer.set(chunk, offset);
        offset += chunk.length;
      }

      // Extract exactly 1 second (discard excess)
      const oneSecondBuffer = fullBuffer.slice(0, samplesPerSecond);

      // Transfer to worker for analysis (transferable for zero-copy)
      worker.postMessage(
        {
          type: 'analyze',
          pcm: oneSecondBuffer,
          sampleRate,
          fftSize: 1024,
          timestamp: Date.now()
        },
        [oneSecondBuffer.buffer]
      );

      // Reset accumulator, keeping excess samples
      const excess = totalSamples - samplesPerSecond;
      if (excess > 0) {
        const excessBuffer = fullBuffer.slice(samplesPerSecond);
        accumulatedSamples = [excessBuffer];
        totalSamples = excess;
      } else {
        accumulatedSamples = [];
        totalSamples = 0;
      }
    }
  };

  // Audio processing node and source
  let audioNode: AudioWorkletNode | ScriptProcessorNode;
  let sourceNode: MediaStreamAudioSourceNode;

  // Try AudioWorklet first (preferred, low-latency)
  const useAudioWorklet = 'audioWorklet' in audioContext;

  if (useAudioWorklet) {
    try {
      // Load AudioWorklet processor
      await audioContext.audioWorklet.addModule('/audio-processor.js');

      // Create AudioWorklet node
      const workletNode = new AudioWorkletNode(
        audioContext,
        'audio-capture-processor'
      );

      // Listen for PCM chunks from worklet
      workletNode.port.onmessage = (event) => {
        const { audioChunk } = event.data;
        if (audioChunk && audioChunk.length > 0) {
          accumulatedSamples.push(new Float32Array(audioChunk));
          totalSamples += audioChunk.length;
          processAccumulatedAudio();
        }
      };

      // Connect audio graph
      sourceNode = audioContext.createMediaStreamSource(stream);
      sourceNode.connect(workletNode);
      // Don't connect to destination (we don't want playback)

      audioNode = workletNode;

      console.log('✓ Using AudioWorklet for live audio capture');
    } catch (error) {
      console.warn('AudioWorklet failed, falling back to ScriptProcessor:', error);
      // Fall through to ScriptProcessor
    }
  }

  // Fallback: ScriptProcessorNode (deprecated but widely supported)
  if (!audioNode!) {
    const bufferSize = 4096; // Must be power of 2: 256, 512, 1024, 2048, 4096, 8192, 16384
    const scriptNode = audioContext.createScriptProcessor(
      bufferSize,
      1, // mono input
      1  // mono output
    );

    scriptNode.onaudioprocess = (event) => {
      const inputBuffer = event.inputBuffer;
      const channelData = inputBuffer.getChannelData(0);
      
      // Copy to avoid detached buffer issues
      const chunk = new Float32Array(channelData);
      accumulatedSamples.push(chunk);
      totalSamples += chunk.length;
      processAccumulatedAudio();
    };

    // Connect audio graph
    sourceNode = audioContext.createMediaStreamSource(stream);
    sourceNode.connect(scriptNode);
    scriptNode.connect(audioContext.destination); // Required for ScriptProcessor to run

    audioNode = scriptNode;

    console.log('✓ Using ScriptProcessorNode for live audio capture (fallback)');
  }

  // Return controller for stopping analysis
  return {
    stop: async () => {
      console.log('Stopping live audio analysis...');

      // Disconnect audio nodes
      if (sourceNode) {
        sourceNode.disconnect();
      }
      if (audioNode) {
        audioNode.disconnect();
      }

      // Stop all media stream tracks
      stream.getTracks().forEach((track) => track.stop());

      // Close audio context
      if (audioContext.state !== 'closed') {
        await audioContext.close();
      }

      // Terminate worker
      worker.terminate();
      URL.revokeObjectURL(workerUrl);

      console.log('✓ Live audio analysis stopped and cleaned up');
    }
  };
}

/**
 * Generate worker code as string for blob creation
 * This embeds the worker logic inline to avoid separate file loading issues
 */
function computeDominantFrequencyWorkerCode(): string {
  return `
    // ===== FFTfast implementation (copied for worker context) =====
    class FFTfast {
      constructor(size) {
        if (size <= 0 || (size & (size - 1)) !== 0) {
          throw new Error('FFT size must be a positive power of 2');
        }
        this.size = size;
        this.cosTable = new Float32Array(size / 2);
        this.sinTable = new Float32Array(size / 2);
        this.reverseTable = new Uint32Array(size);
        
        for (let i = 0; i < size / 2; i++) {
          const angle = (2 * Math.PI * i) / size;
          this.cosTable[i] = Math.cos(angle);
          this.sinTable[i] = Math.sin(angle);
        }
        
        for (let i = 0; i < size; i++) {
          this.reverseTable[i] = this.reverseBits(i);
        }
      }
      
      reverseBits(index) {
        let reversed = 0;
        let numBits = Math.log2(this.size);
        for (let i = 0; i < numBits; i++) {
          reversed = (reversed << 1) | (index & 1);
          index >>= 1;
        }
        return reversed;
      }
      
      forward(input) {
        if (input.length !== this.size) {
          throw new Error('Input size must match FFT size');
        }
        const real = new Float32Array(this.size);
        const imag = new Float32Array(this.size);
        const magnitude = new Float32Array(this.size);
        
        for (let i = 0; i < this.size; i++) {
          real[i] = input[this.reverseTable[i]];
        }
        
        for (let size = 2; size <= this.size; size *= 2) {
          const halfSize = size / 2;
          const tableStep = this.size / size;
          for (let i = 0; i < this.size; i += size) {
            for (let j = 0; j < halfSize; j++) {
              const twiddle = j * tableStep;
              const evenIndex = i + j;
              const oddIndex = i + j + halfSize;
              const evenReal = real[evenIndex];
              const evenImag = imag[evenIndex];
              const oddReal = real[oddIndex];
              const oddImag = imag[oddIndex];
              const wReal = this.cosTable[twiddle];
              const wImag = -this.sinTable[twiddle];
              const tempReal = oddReal * wReal - oddImag * wImag;
              const tempImag = oddReal * wImag + oddImag * wReal;
              real[oddIndex] = evenReal - tempReal;
              imag[oddIndex] = evenImag - tempImag;
              real[evenIndex] = evenReal + tempReal;
              imag[evenIndex] = evenImag + tempImag;
            }
          }
        }
        
        for (let i = 0; i < this.size; i++) {
          magnitude[i] = Math.sqrt(real[i] * real[i] + imag[i] * imag[i]);
        }
        return magnitude;
      }
    }
    
    // ===== Frequency calculation functions =====
    function applyHannWindow(frame) {
      const windowed = new Float32Array(frame.length);
      for (let i = 0; i < frame.length; i++) {
        const windowValue = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (frame.length - 1)));
        windowed[i] = frame[i] * windowValue;
      }
      return windowed;
    }
    
    function findPeakBin(spectrum, sampleRate, fftSize, minFreq = 20, maxFreq = 8000) {
      const freqPerBin = sampleRate / fftSize;
      const minBin = Math.floor(minFreq / freqPerBin);
      const maxBin = Math.min(Math.ceil(maxFreq / freqPerBin), spectrum.length - 1);
      let peakBin = minBin;
      let peakValue = spectrum[minBin];
      for (let i = minBin + 1; i <= maxBin; i++) {
        if (spectrum[i] > peakValue) {
          peakValue = spectrum[i];
          peakBin = i;
        }
      }
      return { binIndex: peakBin, value: peakValue };
    }
    
    function computeDominantFrequency(pcm, sampleRate, fftSize = 1024, hopLength = 512) {
      if (fftSize <= 0 || (fftSize & (fftSize - 1)) !== 0) {
        throw new Error('FFT size must be a power of 2');
      }
      if (pcm.length < fftSize) {
        return { peakFreq: 0, peakVal: 0 };
      }
      const fft = new FFTfast(fftSize);
      const numBins = fftSize / 2 + 1;
      const accumulatedSpectrum = new Float32Array(numBins);
      let numFrames = 0;
      for (let i = 0; i <= pcm.length - fftSize; i += hopLength) {
        const frame = pcm.slice(i, i + fftSize);
        const windowedFrame = applyHannWindow(frame);
        const magnitude = fft.forward(windowedFrame);
        for (let j = 0; j < numBins; j++) {
          accumulatedSpectrum[j] += magnitude[j];
        }
        numFrames++;
      }
      if (numFrames > 0) {
        for (let i = 0; i < numBins; i++) {
          accumulatedSpectrum[i] /= numFrames;
        }
      }
      const { binIndex, value } = findPeakBin(accumulatedSpectrum, sampleRate, fftSize, 20, 8000);
      const peakFreq = (binIndex * sampleRate) / fftSize;
      return { peakFreq, peakVal: value };
    }
    
    // ===== Worker message handler =====
    self.onmessage = (event) => {
      const { type, pcm, sampleRate, fftSize = 1024, timestamp } = event.data;
      if (type === 'analyze') {
        try {
          const result = computeDominantFrequency(pcm, sampleRate, fftSize);
          self.postMessage({
            type: 'result',
            peakFreq: result.peakFreq,
            peakVal: result.peakVal,
            timestamp
          });
        } catch (error) {
          console.error('Worker error:', error);
          self.postMessage({ type: 'result', peakFreq: 0, peakVal: 0, timestamp });
        }
      }
    };
    
    self.postMessage({ type: 'ready' });
  `;
}
