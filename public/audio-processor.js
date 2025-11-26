/**
 * AudioWorkletProcessor for real-time PCM audio capture
 * Runs on audio rendering thread for minimal latency
 * Posts mono Float32Array chunks to main thread without processing
 */
class AudioCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.chunkSize = 128; // Small chunks for low latency
  }

  /**
   * Process audio frames - called per render quantum (128 samples default)
   * Keep this minimal - no heavy computation on audio thread
   */
  process(inputs, outputs, parameters) {
    const input = inputs[0];
    
    // Handle multi-channel input by converting to mono
    if (input && input.length > 0) {
      const channel0 = input[0];
      
      // If stereo or multi-channel, average to mono
      if (input.length > 1) {
        const mono = new Float32Array(channel0.length);
        for (let i = 0; i < channel0.length; i++) {
          let sum = 0;
          for (let ch = 0; ch < input.length; ch++) {
            sum += input[ch][i];
          }
          mono[i] = sum / input.length;
        }
        // Transfer ownership to avoid copy
        this.port.postMessage({ audioChunk: mono }, [mono.buffer]);
      } else {
        // Already mono, copy and transfer
        const mono = new Float32Array(channel0);
        this.port.postMessage({ audioChunk: mono }, [mono.buffer]);
      }
    }
    
    // Return true to keep processor alive
    return true;
  }
}

registerProcessor('audio-capture-processor', AudioCaptureProcessor);
