/**
 * AudioWorklet processor - forwards 16 kHz mono PCM chunks to the main thread.
 * The page creates AudioContext({ sampleRate: 16000 }), so browser resampling
 * happens before this processor sees the samples.
 */
class AudioCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._size = 1280; // 80 ms at 16 kHz, matching /transcribe-streaming guidance.
    this._buf = new Float32Array(this._size + 128); // room for one extra quantum
    this._pos = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const samples = input[0]; // 128 Float32 samples at 48 kHz

    this._buf.set(samples, this._pos);
    this._pos += samples.length;

    if (this._pos >= this._size) {
      this.port.postMessage({
        type: 'audio',
        samples: new Float32Array(this._buf.subarray(0, this._size)),
      });
      // carry leftover into next chunk
      const leftover = this._pos - this._size;
      if (leftover > 0) {
        this._buf.copyWithin(0, this._size, this._pos);
      }
      this._pos = leftover;
    }

    return true;
  }
}

registerProcessor('audio-capture-processor', AudioCaptureProcessor);
