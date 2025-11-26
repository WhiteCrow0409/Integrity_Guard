import { FFTfast } from './FFTfast';

export class GaussianDFTAudioDetectorFast {
  private frameLength: number;
  private hopLength: number;
  private sampleRate: number;
  private fft: FFTfast;
  private minEnergyThreshold: number;
  private minFrequency: number;
  private maxFrequency: number;

  constructor(frameLength = 1024, hopLength = 512, sampleRate = 16000, minEnergyThreshold = 0.002, minFrequency = 80, maxFrequency = 3400) {
    this.frameLength = frameLength;
    this.hopLength = hopLength;
    this.sampleRate = sampleRate;
    this.fft = new FFTfast(frameLength);
    this.minEnergyThreshold = minEnergyThreshold; // Reduced default threshold
    this.minFrequency = minFrequency; // Expanded frequency range
    this.maxFrequency = maxFrequency; // Expanded frequency range
  }

  private async getAudioData(audioData: ArrayBuffer): Promise<Float32Array> {
    try {
      const audioContext = new AudioContext({ sampleRate: this.sampleRate });
      const audioBuffer = await audioContext.decodeAudioData(audioData);
      return audioBuffer.getChannelData(0);
    } catch (error) {
      throw new Error('Failed to decode audio data. Please ensure it\'s valid audio format.');
    }
  }

  private calculateFrameEnergy(frame: Float32Array): number {
    let energy = 0;
    for (let i = 0; i < frame.length; i++) {
      energy += frame[i] * frame[i];
    }
    return energy / frame.length;
  }

  private isInSpeechFrequencyRange(frequencyIndex: number): boolean {
    const frequency = (frequencyIndex * this.sampleRate) / this.frameLength;
    return frequency >= this.minFrequency && frequency <= this.maxFrequency;
  }

  private frameSignal(signal: Float32Array): Float32Array[] {
    if (signal.length < this.frameLength) {
      throw new Error('Audio data is too short for analysis.');
    }

    const frames: Float32Array[] = [];
    for (let i = 0; i < signal.length - this.frameLength; i += this.hopLength) {
      const frame = signal.slice(i, i + this.frameLength);
      frames.push(frame);
    }
    return frames;
  }

  private applyWindow(frame: Float32Array): Float32Array {
    // Hanning window
    const window = new Float32Array(frame.length);
    for (let i = 0; i < frame.length; i++) {
      window[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (frame.length - 1)));
    }
    return frame.map((x, i) => x * window[i]);
  }

  private extractDFTFeatures(frames: Float32Array[]): Float32Array[] {
    return frames.map(frame => {
      const windowedFrame = this.applyWindow(frame);
      const dft = this.fft.forward(windowedFrame);
      return new Float32Array(dft.slice(0, this.frameLength / 2 + 1));
    });
  }

  private estimateNoiseVariance(features: Float32Array[], nNoiseFrames = 10): Float32Array {
    const numFramesToUse = Math.min(features.length, nNoiseFrames);
    if (numFramesToUse === 0) {
      throw new Error('Not enough frames for noise estimation.');
    }

    const variance = new Float32Array(features[0].length);
    
    for (let i = 0; i < numFramesToUse; i++) {
      for (let j = 0; j < variance.length; j++) {
        variance[j] += Math.pow(features[i][j], 2);
      }
    }

    return variance.map(v => v / numFramesToUse);
  }

  private estimatePrioriSNR(features: Float32Array[], sigmaN: Float32Array, alpha = 0.98): Float32Array[] {
    const xi: Float32Array[] = [];
    
    // Initial estimate
    const firstFrame = features[0];
    const gamma = firstFrame.map((y, j) => Math.pow(y, 2) / (sigmaN[j] || Number.EPSILON));
    xi.push(new Float32Array(gamma.map(g => Math.max(g - 1, 0))));

    // Estimate for remaining frames
    for (let m = 1; m < features.length; m++) {
      const frame = features[m];
      const prevXi = xi[m - 1];
      const newXi = new Float32Array(frame.length);

      for (let j = 0; j < frame.length; j++) {
        const sigmaNJ = sigmaN[j] || Number.EPSILON;
        const ampPrev = Math.sqrt(prevXi[j] * sigmaNJ);
        const gammaCurrent = Math.pow(frame[j], 2) / sigmaNJ;
        newXi[j] = alpha * (Math.pow(ampPrev, 2) / sigmaNJ) + 
                   (1 - alpha) * Math.max(gammaCurrent - 1, 0);
      }

      xi.push(newXi);
    }

    return xi;
  }

  public async detectSpeechFromArrayBuffer(audioData: ArrayBuffer, threshold = 0.5): Promise<{
    speechFrames: boolean[];
    logLikelihood: number[];
    speechPercentage: number;
    totalFrames: number;
    duration: number;
  }> {
    const audio = await this.getAudioData(audioData);
    const frames = this.frameSignal(audio);
    const features = this.extractDFTFeatures(frames);
    
    const sigmaN = this.estimateNoiseVariance(features);
    const xi = this.estimatePrioriSNR(features, sigmaN);
    
    const logLikelihood: number[] = [];
    const speechFrames: boolean[] = [];

    for (let m = 0; m < features.length; m++) {
      const frame = features[m];
      const rawFrame = frames[m];
      
      // Calculate frame energy first
      const frameEnergy = this.calculateFrameEnergy(rawFrame);
      
      // Skip frames with insufficient energy
      if (frameEnergy < this.minEnergyThreshold) {
        logLikelihood.push(-10); // Low likelihood for low energy frames
        speechFrames.push(false);
        continue;
      }
      
      let sumLogL = 0;
      let speechFreqCount = 0;
      let totalSpeechFreqEnergy = 0;

      for (let j = 0; j < frame.length; j++) {
        // Only consider frequencies in speech range
        if (this.isInSpeechFrequencyRange(j)) {
          const sigmaS = xi[m][j] * sigmaN[j];
          const y2 = Math.pow(frame[j], 2);
          const gamma = y2 / (sigmaN[j] || Number.EPSILON);
          const xiJ = sigmaS / (sigmaN[j] || Number.EPSILON);
          
          const L = (1 / (1 + xiJ)) * Math.exp((gamma * xiJ) / (1 + xiJ));
          sumLogL += Math.log(Math.max(L, Number.EPSILON));
          
          speechFreqCount++;
          totalSpeechFreqEnergy += y2;
        }
      }

      // Calculate likelihood only if we have speech frequencies
      const avgLogL = speechFreqCount > 0 ? sumLogL / speechFreqCount : -10;
      const avgSpeechEnergy = speechFreqCount > 0 ? totalSpeechFreqEnergy / speechFreqCount : 0;
      
      logLikelihood.push(avgLogL);
      
      // Enhanced detection: require both likelihood threshold and sufficient energy in speech frequencies
      const isSpeech = avgLogL > Math.log(threshold) && 
                       avgSpeechEnergy > this.minEnergyThreshold * 3 && // Reduced from 10x to 3x energy threshold for speech frequencies
                       frameEnergy > this.minEnergyThreshold;
      
      speechFrames.push(isSpeech);
    }

    const speechPercentage = (speechFrames.filter(Boolean).length / speechFrames.length) * 100;
    const duration = (audio.length / this.sampleRate);

    return { 
      speechFrames, 
      logLikelihood, 
      speechPercentage,
      totalFrames: speechFrames.length,
      duration
    };
  }

  public async detectSpeech(file: File, threshold = 0.5): Promise<{
    speechFrames: boolean[];
    logLikelihood: number[];
    speechPercentage: number;
    totalFrames: number;
    duration: number;
  }> {
    if (!file.type.startsWith('audio/')) {
      throw new Error('Please provide a valid audio file.');
    }

    const audioData = await file.arrayBuffer();
    return this.detectSpeechFromArrayBuffer(audioData, threshold);
  }
}