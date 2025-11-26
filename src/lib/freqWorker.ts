/**
 * Web Worker for frequency analysis
 * Receives 1-second PCM chunks and computes dominant frequency
 * Runs FFT computation off main thread to prevent UI blocking
 */

import { computeDominantFrequency } from './freqCalculator';

/**
 * Message format received from main thread
 */
interface WorkerInput {
  type: 'analyze';
  pcm: Float32Array;
  sampleRate: number;
  fftSize?: number;
  timestamp: number;
}

/**
 * Message format sent back to main thread
 */
interface WorkerOutput {
  type: 'result';
  peakFreq: number;
  peakVal: number;
  timestamp: number;
}

/**
 * Worker message handler
 */
self.onmessage = (event: MessageEvent<WorkerInput>) => {
  const { type, pcm, sampleRate, fftSize = 1024, timestamp } = event.data;

  if (type === 'analyze') {
    try {
      // Compute dominant frequency using FFT
      const result = computeDominantFrequency(pcm, sampleRate, fftSize);

      // Send result back to main thread
      const response: WorkerOutput = {
        type: 'result',
        peakFreq: result.peakFreq,
        peakVal: result.peakVal,
        timestamp
      };

      self.postMessage(response);
    } catch (error) {
      console.error('Worker frequency analysis error:', error);
      // Send zero frequency on error
      self.postMessage({
        type: 'result',
        peakFreq: 0,
        peakVal: 0,
        timestamp
      });
    }
  }
};

// Indicate worker is ready
self.postMessage({ type: 'ready' });
