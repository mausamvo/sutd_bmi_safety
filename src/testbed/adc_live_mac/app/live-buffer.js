// Browser-side sample buffers for the live view and for lossless captures.
// (From the macOS-optimised adc-live: bounded display history, plot
// decimation that keeps spikes, and gap-checked capture buffers.)
// Display history is bounded and independent of lossless capture storage.
export const MAX_DISPLAY_SAMPLES = 60000;
export const MAX_PLOT_POINTS = 600;
export const DISPLAY_INTERVAL_MS = 50;
export const MAX_CAPTURE_SAMPLES = 120000;

export class SampleBuffer {
  constructor(capacity = MAX_DISPLAY_SAMPLES) {
    this.capacity = capacity;
    this.values = new Array(capacity);
    this.start = 0;
    this.size = 0;
  }

  push(sample) {
    this.values[(this.start + this.size) % this.capacity] = sample;
    if (this.size < this.capacity) this.size += 1;
    else this.start = (this.start + 1) % this.capacity;
  }

  latest() {
    return this.size ? this.values[(this.start + this.size - 1) % this.capacity] : null;
  }

  recent(cutoff) {
    const result = [];
    for (let i = this.size - 1; i >= 0; i -= 1) {
      const sample = this.values[(this.start + i) % this.capacity];
      if (sample.timestamp < cutoff) break;
      result.push(sample);
    }
    return result.reverse();
  }
}

// Keep local minima/maxima for both signals on every channel. Subsampling is
// for plots only: CSV rows never pass through this function.
export function decimateForDisplay(samples, limit = MAX_PLOT_POINTS) {
  if (samples.length <= limit) return samples;
  const series = Array.from({ length: 4 }, (_, ch) =>
    ["a", "e"].map((field) => [ch, field])
  ).flat();
  const buckets = Math.max(1, Math.floor((limit - 2) / (series.length * 2)));
  const selected = new Set([0, samples.length - 1]);
  for (let bucket = 0; bucket < buckets; bucket += 1) {
    const start = Math.floor((bucket * samples.length) / buckets);
    const end = Math.floor(((bucket + 1) * samples.length) / buckets);
    for (const [ch, field] of series) {
      let minIndex = start;
      let maxIndex = start;
      let min = Infinity;
      let max = -Infinity;
      for (let i = start; i < end; i += 1) {
        const value = samples[i][`ch${ch}`]?.[field] ?? 0;
        if (value < min) { min = value; minIndex = i; }
        if (value > max) { max = value; maxIndex = i; }
      }
      selected.add(minIndex);
      selected.add(maxIndex);
    }
  }
  return [...selected].sort((a, b) => a - b).map((i) => samples[i]);
}

export class CaptureBuffer {
  constructor(startedAt, durationMs, limit = MAX_CAPTURE_SAMPLES) {
    this.startedAt = startedAt;
    this.endsAt = startedAt + durationMs;
    this.limit = limit;
    this.samples = [];
    this.error = null;
    this.previous = null;
    this.complete = false;
  }

  fail(message) {
    if (!this.error) this.error = new Error(message);
  }

  push(sample) {
    if (sample.timestamp < this.startedAt || this.complete || this.error) return;
    if (this.previous && Number.isInteger(sample.sequence) &&
        Number.isInteger(this.previous.sequence) &&
        (sample.sequence !== this.previous.sequence + 1 ||
         sample.streamId !== this.previous.streamId)) {
      this.fail("Samples were lost or the stream restarted. Capture was not saved; please retry.");
      return;
    }
    // A later sample is the stream boundary marker. Seeing it confirms all
    // preceding ordered batches arrived before we finalize the capture.
    if (sample.timestamp >= this.endsAt) {
      this.complete = true;
      return;
    }
    if (this.samples.length >= this.limit) {
      this.fail("Capture exceeded the safe memory limit. Use a shorter capture duration.");
      return;
    }
    this.samples.push(sample);
    this.previous = sample;
  }

  finish(requiredChannels = 0) {
    if (this.error) throw this.error;
    if (!this.samples.length) throw new Error("No samples received. Check the device connection and try again.");
    if (!this.complete) throw new Error("The sample stream stalled before capture completed. Capture was not saved; please retry.");
    for (const sample of this.samples) {
      for (let channel = 0; channel < requiredChannels; channel += 1) {
        const values = sample[`ch${channel}`];
        if (!Number.isFinite(values?.a) || !Number.isFinite(values?.e)) {
          throw new Error(`Channel ${channel} is missing or invalid. Capture was not saved; check the selected channel count and device protocol.`);
        }
      }
    }
    return this.samples;
  }
}

export function waitFor(ms, signal) {
  return new Promise((resolve, reject) => {
    const aborted = () => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", aborted);
      reject(new DOMException("Recording cancelled", "AbortError"));
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", aborted);
      resolve();
    }, ms);
    if (signal?.aborted) aborted();
    else signal?.addEventListener("abort", aborted, { once: true });
  });
}

export function csvCell(value) {
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}
