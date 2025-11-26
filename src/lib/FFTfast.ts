export class FFTfast {
  private size: number;
  private cosTable: Float32Array;
  private sinTable: Float32Array;
  private reverseTable: Uint32Array;

  constructor(size: number) {
    if (size <= 0 || (size & (size - 1)) !== 0) {
      throw new Error('FFT size must be a positive power of 2');
    }

    this.size = size;
    this.cosTable = new Float32Array(size / 2);
    this.sinTable = new Float32Array(size / 2);
    this.reverseTable = new Uint32Array(size);

    // Precompute twiddle factors
    for (let i = 0; i < size / 2; i++) {
      const angle = (2 * Math.PI * i) / size;
      this.cosTable[i] = Math.cos(angle);
      this.sinTable[i] = Math.sin(angle);
    }

    // Precompute bit-reversed indices
    for (let i = 0; i < size; i++) {
      this.reverseTable[i] = this.reverseBits(i);
    }
  }

  private reverseBits(index: number): number {
    let reversed = 0;
    let numBits = Math.log2(this.size);
    
    for (let i = 0; i < numBits; i++) {
      reversed = (reversed << 1) | (index & 1);
      index >>= 1;
    }
    
    return reversed;
  }

  public forward(input: Float32Array): Float32Array {
    if (input.length !== this.size) {
      throw new Error(`Input size must match FFT size (${this.size})`);
    }

    // Create output arrays
    const real = new Float32Array(this.size);
    const imag = new Float32Array(this.size);
    const magnitude = new Float32Array(this.size);

    // Copy input data to real array in bit-reversed order
    for (let i = 0; i < this.size; i++) {
      real[i] = input[this.reverseTable[i]];
    }

    // Perform radix-2 FFT
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
          
          // Use precomputed twiddle factors
          const wReal = this.cosTable[twiddle];
          const wImag = -this.sinTable[twiddle];
          
          // Butterfly operation
          const tempReal = oddReal * wReal - oddImag * wImag;
          const tempImag = oddReal * wImag + oddImag * wReal;
          
          real[oddIndex] = evenReal - tempReal;
          imag[oddIndex] = evenImag - tempImag;
          real[evenIndex] = evenReal + tempReal;
          imag[evenIndex] = evenImag + tempImag;
        }
      }
    }

    // Calculate magnitude
    for (let i = 0; i < this.size; i++) {
      magnitude[i] = Math.sqrt(real[i] * real[i] + imag[i] * imag[i]);
    }

    return magnitude;
  }

  // Optional: Implement inverse FFT
  public inverse(realInput: Float32Array, imagInput: Float32Array): Float32Array {
    if (realInput.length !== this.size || (imagInput && imagInput.length !== this.size)) {
      throw new Error(`Input size must match FFT size (${this.size})`);
    }
    
    const real = new Float32Array(this.size);
    const imag = new Float32Array(this.size);
    const result = new Float32Array(this.size);
    
    // Copy input with bit reversal
    for (let i = 0; i < this.size; i++) {
      real[i] = realInput[this.reverseTable[i]];
      imag[i] = imagInput ? -imagInput[this.reverseTable[i]] : 0;
    }
    
    // Perform radix-2 FFT (similar to forward, but with conjugated twiddle factors)
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
          
          // Use precomputed twiddle factors but with positive imaginary part
          const wReal = this.cosTable[twiddle];
          const wImag = this.sinTable[twiddle]; // Note: positive for inverse
          
          const tempReal = oddReal * wReal - oddImag * wImag;
          const tempImag = oddReal * wImag + oddImag * wReal;
          
          real[oddIndex] = evenReal - tempReal;
          imag[oddIndex] = evenImag - tempImag;
          real[evenIndex] = evenReal + tempReal;
          imag[evenIndex] = evenImag + tempImag;
        }
      }
    }
    
    // Scale and copy to result
    const scale = 1.0 / this.size;
    for (let i = 0; i < this.size; i++) {
      result[i] = real[i] * scale;
    }
    
    return result;
  }
}