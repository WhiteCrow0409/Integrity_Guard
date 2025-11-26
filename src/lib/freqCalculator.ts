import { FFTfast } from './FFTfast';

/**
 * Result from frequency analysis
 */
export interface FrequencyAnalysisResult {
  peakFreq: number;      // Dominant frequency in Hz
  peakVal: number;       // Magnitude at peak
  spectrum: Float32Array; // Full magnitude spectrum
}

/**
 * Applies Hann window to reduce spectral leakage
 * @param frame - Input PCM samples
 * @returns Windowed samples
 */
function applyHannWindow(frame: Float32Array): Float32Array {
  const windowed = new Float32Array(frame.length);
  for (let i = 0; i < frame.length; i++) {
    const windowValue = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (frame.length - 1)));
    windowed[i] = frame[i] * windowValue;
  }
  return windowed;
}

/**
 * Find the index and value of the maximum element in array
 * Optionally restricts search to a frequency range
 */
function findPeakBin(
  spectrum: Float32Array,
  sampleRate: number,
  fftSize: number,
  minFreq = 20,    // Human audible range lower bound
  maxFreq = 8000   // Restrict to mid-high range to avoid DC/noise
): { binIndex: number; value: number } {
  const freqPerBin = sampleRate / fftSize;
  const minBin = Math.floor(minFreq / freqPerBin);
  const maxBin = Math.min(
    Math.ceil(maxFreq / freqPerBin),
    spectrum.length - 1
  );

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

/**
 * Compute dominant frequency from PCM audio chunk using FFT
 * 
 * Strategy: For a 1-second chunk, we compute FFT on overlapping windows
 * and average the magnitude spectra to find the most consistent peak.
 * This reduces noise and provides stable frequency detection.
 * 
 * @param pcm - Raw PCM audio samples (Float32Array)
 * @param sampleRate - Sample rate in Hz (e.g., 48000)
 * @param fftSize - FFT window size, must be power of 2 (default 1024)
 * @param hopLength - Hop size between windows (default 512, 50% overlap)
 * @returns Frequency analysis result with peak frequency and magnitude
 */
export function computeDominantFrequency(
  pcm: Float32Array,
  sampleRate: number,
  fftSize = 1024,
  hopLength = 512
): FrequencyAnalysisResult {
  // Ensure FFT size is valid
  if (fftSize <= 0 || (fftSize & (fftSize - 1)) !== 0) {
    throw new Error('FFT size must be a power of 2');
  }

  if (pcm.length < fftSize) {
    // Not enough samples, return zero frequency
    return {
      peakFreq: 0,
      peakVal: 0,
      spectrum: new Float32Array(fftSize / 2 + 1)
    };
  }

  const fft = new FFTfast(fftSize);
  const numBins = fftSize / 2 + 1;
  const accumulatedSpectrum = new Float32Array(numBins);
  let numFrames = 0;

  // Process overlapping windows
  for (let i = 0; i <= pcm.length - fftSize; i += hopLength) {
    const frame = pcm.slice(i, i + fftSize);
    const windowedFrame = applyHannWindow(frame);
    
    // Compute FFT magnitude spectrum
    const magnitude = fft.forward(windowedFrame);
    
    // Accumulate spectrum (only first half + Nyquist)
    for (let j = 0; j < numBins; j++) {
      accumulatedSpectrum[j] += magnitude[j];
    }
    numFrames++;
  }

  // Average the accumulated spectrum
  if (numFrames > 0) {
    for (let i = 0; i < numBins; i++) {
      accumulatedSpectrum[i] /= numFrames;
    }
  }

  // Find peak frequency in human-audible range
  const { binIndex, value } = findPeakBin(
    accumulatedSpectrum,
    sampleRate,
    fftSize,
    20,   // Min: 20 Hz (below human hearing)
    8000  // Max: 8 kHz (catches speech/music fundamentals)
  );

  // Convert bin index to frequency
  const peakFreq = (binIndex * sampleRate) / fftSize;

  return {
    peakFreq,
    peakVal: value,
    spectrum: accumulatedSpectrum
  };
}
