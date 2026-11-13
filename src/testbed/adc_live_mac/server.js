// server.js: Next.js page + Socket.IO + serial board, optimised for macOS.
//
// Samples go to every connected client (the browser page and the Python live
// prediction sidecar, `python live.py serve`) in acknowledged batches with a
// sequence number, so a capture can tell when samples were lost.
// The Live tab's messages (live_cmd / live_prediction / live_status) are
// relayed between the browser and the sidecar.
const path = require('node:path');
const { createServer } = require('node:http');
const { randomUUID } = require('node:crypto');
const next = require('next');
const { Server } = require('socket.io');
const connect = require('./readSerial');

const APP_DIR = __dirname;
const SEMG_DIR = path.resolve(APP_DIR, '..', '..', '..');
// Recordings go to new/data/recordings/<Save Subfolder>/ unless SAVE_ROOT is set.
const SAVE_ROOT = path.resolve(process.env.SAVE_ROOT || path.join(APP_DIR, '..', '..', 'data', 'recordings'));
const relativeRoot = path.relative(SEMG_DIR, SAVE_ROOT);
const SAVE_ROOT_LABEL = relativeRoot && !relativeRoot.startsWith('..') && !path.isAbsolute(relativeRoot)
  ? relativeRoot.split(path.sep).join('/') : SAVE_ROOT;
// The save route (app/api/save-capture) and live.py read these.
process.env.SAVE_ROOT = SAVE_ROOT;
process.env.SAVE_ROOT_LABEL = SAVE_ROOT_LABEL;

const LIVE_EVENTS = ['live_cmd', 'live_prediction', 'live_status'];
const MAX_LIVE_MESSAGE_BYTES = 256 * 1024;

function positiveInteger(name, fallback, maximum) {
  const value = process.env[name] === undefined ? fallback : Number(process.env[name]);
  if (!Number.isInteger(value) || value < 1 || value > maximum) {
    throw new Error(`${name} must be an integer from 1 to ${maximum}.`);
  }
  return value;
}

// Reliable, acknowledged batches with a bounded per-client backlog. If a
// client cannot keep up, its stream is ended visibly instead of silently
// dropping samples or letting buffers grow without limit.
function createSampleStream(options = {}) {
  const batchInterval = options.batchInterval ?? 20;
  const maxBatchSize = options.maxBatchSize ?? 250;
  const maxQueuedSamples = options.maxQueuedSamples ?? 2500;
  const ackTimeout = options.ackTimeout ?? 5000;
  const streamId = options.streamId || randomUUID();
  const clients = new Set();
  let pending = [];
  let sequence = 0;

  function remove(client) {
    clearTimeout(client.timer);
    client.queue.length = 0;
    client.closed = true;
    clients.delete(client);
  }

  function fail(client, message) {
    remove(client);
    client.socket.emit('serial_status', { state: 'error', message });
    client.socket.disconnect(true);
  }

  function send(client) {
    if (client.closed || client.inFlight || !client.queue.length) return;
    const samples = client.queue.splice(0, maxBatchSize);
    client.inFlight = true;
    let acknowledged = false;
    client.timer = setTimeout(() => fail(client,
      'Browser did not acknowledge samples. Recording interrupted; reconnecting.'), ackTimeout);
    client.socket.emit('adc_batch', samples, () => {
      if (acknowledged || client.closed) return;
      acknowledged = true;
      clearTimeout(client.timer);
      client.inFlight = false;
      send(client);
    });
  }

  function flush() {
    if (!pending.length) return;
    const samples = pending;
    pending = [];
    for (const client of clients) {
      if (client.queue.length + samples.length > maxQueuedSamples) {
        fail(client, 'Browser fell behind the live stream. Recording interrupted; reconnecting.');
        continue;
      }
      client.queue.push(...samples);
      send(client);
    }
  }

  const timer = setInterval(flush, batchInterval);
  return {
    add(sample) {
      sequence += 1;
      pending.push({ ...sample, sequence, streamId });
      if (pending.length >= maxBatchSize) flush();
    },
    skip() { sequence += 1; }, // marks a gap (bad row, reconnect)
    addClient(socket) {
      flush(); // a new client does not get samples from before it joined
      const client = { socket, queue: [], inFlight: false, closed: false, timer: null };
      clients.add(client);
      socket.on('disconnect', () => remove(client));
    },
    flush,
    stop() {
      clearInterval(timer);
      pending = [];
      for (const client of clients) remove(client);
    },
  };
}

// Optional 50 Hz mains-hum filter (a notch: removes a narrow band around 50 Hz,
// keeps everything else). Applied to the activation values (Act) only: the
// envelope (Env) is computed on the board before the data reaches this Mac,
// so it cannot be cleaned here. The filter always runs, so switching it on has
// no start-up glitch; when it is off, samples are passed on exactly as received.
// The sample rate is measured from the incoming rows (~335/s from the board).
function createNotch50({ frequency = 50, q = 20, defaultRate = 335 } = {}) {
  const state = Array.from({ length: 4 }, () => ({ x1: null, x2: 0, y1: 0, y2: 0 }));
  const times = [];
  let head = 0;
  let rate = defaultRate;
  let coeffs = null;
  let coeffRate = 0;

  function design(fs) {
    if (fs <= 2.2 * frequency) return null; // 50 Hz not representable: pass through
    const w0 = (2 * Math.PI * frequency) / fs;
    const alpha = Math.sin(w0) / (2 * q);
    const cos = Math.cos(w0);
    const a0 = 1 + alpha;
    return { b0: 1 / a0, b1: (-2 * cos) / a0, b2: 1 / a0, a1: (-2 * cos) / a0, a2: (1 - alpha) / a0 };
  }

  function updateRate(timestamp) {
    times.push(timestamp);
    while (times.length - head > 2 && times[times.length - 1] - times[head] > 5000) head += 1;
    if (head > 4096) { times.splice(0, head); head = 0; }
    const span = times[times.length - 1] - times[head];
    if (span >= 1000) rate = (times.length - head - 1) / (span / 1000);
    if (!coeffRate || Math.abs(rate - coeffRate) / coeffRate > 0.005) {
      coeffs = design(rate);
      coeffRate = rate;
    }
  }

  return {
    // Returns the filtered copy of the sample; the input is not changed.
    process(sample) {
      updateRate(sample.timestamp);
      const out = { ...sample };
      for (let c = 0; c < 4; c += 1) {
        const ch = sample[`ch${c}`];
        if (!ch || !Number.isFinite(ch.a)) continue;
        const st = state[c];
        if (st.x1 === null) { st.x1 = st.x2 = st.y1 = st.y2 = ch.a; } // start settled
        let y = ch.a;
        if (coeffs) {
          y = coeffs.b0 * ch.a + coeffs.b1 * st.x1 + coeffs.b2 * st.x2 - coeffs.a1 * st.y1 - coeffs.a2 * st.y2;
        }
        st.x2 = st.x1; st.x1 = ch.a; st.y2 = st.y1; st.y1 = y;
        out[`ch${c}`] = { a: Math.round(y * 100) / 100, e: ch.e };
      }
      return out;
    },
    rate: () => rate,
  };
}

async function main() {
  process.chdir(APP_DIR); // works even when started from another folder
  const app = next({ dev: process.env.NODE_ENV !== 'production', dir: APP_DIR });
  const handle = app.getRequestHandler();
  const testMode = process.env.TEST_MODE === 'true';
  const replaySpec = process.env.REPLAY;
  const sampleRate = positiveInteger('SAMPLE_RATE', 500, 2000);
  const httpPort = positiveInteger('HTTP_PORT', 3000, 65535);
  const baudRate = positiveInteger('SERIAL_BAUD', 115200, 4000000);
  const expectedChannels = positiveInteger('SERIAL_CHANNELS', 4, 4);
  const host = process.env.HTTP_HOST || '127.0.0.1';
  await app.prepare();
  const server = createServer((req, res) => handle(req, res));
  const io = new Server(server, {
    // No authentication: only this Mac, and only this page, may connect.
    // (The Python sidecar sends no Origin header and is allowed.)
    allowRequest: (request, callback) => {
      const origin = request.headers.origin;
      let sameOrigin = !origin;
      if (origin) {
        try { sameOrigin = new URL(origin).host === request.headers.host; }
        catch { sameOrigin = false; }
      }
      callback(null, sameOrigin);
    },
  });
  const stream = createSampleStream();
  const notch = createNotch50();
  let notchOn = false; // off by default: data exactly as the board sends it
  // Every sample (board, simulation or replay) goes through here.
  const emitSample = (sample) => {
    const filtered = notch.process(sample);
    stream.add(notchOn ? { ...filtered, notch50: true } : sample);
  };
  const filterStatus = () => ({ notch50: notchOn, rateHz: Math.round(notch.rate()) });
  let status = { state: 'connecting', baudRate };
  let reader;
  let simulator;
  let replay;
  let shuttingDown = false;
  const publishStatus = (nextStatus) => {
    // Mark the sequence too, so no capture joins samples across a gap.
    if (nextStatus.state === 'reconnecting' && status.state !== 'reconnecting') stream.skip();
    status = nextStatus;
    io.emit('serial_status', status);
  };

  io.on('connection', (socket) => {
    stream.addClient(socket);
    socket.emit('serial_status', status);
    socket.emit('app_config', { saveRootLabel: SAVE_ROOT_LABEL });
    socket.emit('filter_status', filterStatus());
    socket.on('set_filter', (msg) => {
      if (!msg || typeof msg.notch50 !== 'boolean' || msg.notch50 === notchOn) return;
      notchOn = msg.notch50;
      stream.skip(); // no capture can contain both filtered and unfiltered samples
      console.log(`50 Hz filter ${notchOn ? 'ON' : 'OFF'}`);
      io.emit('filter_status', filterStatus());
    });
    for (const event of LIVE_EVENTS) {
      socket.on(event, (msg) => {
        if (!msg || typeof msg !== 'object' || Array.isArray(msg)) return;
        let size;
        try { size = Buffer.byteLength(JSON.stringify(msg)); } catch { return; }
        if (size > MAX_LIVE_MESSAGE_BYTES) return;
        socket.broadcast.emit(event, msg);
      });
    }
  });

  const shutdown = () => {
    if (shuttingDown) return;
    shuttingDown = true;
    clearInterval(simulator);
    replay?.stop();
    reader?.stop();
    stream.stop();
    // Let in-flight saves finish before closing Next, with a time limit.
    const deadline = setTimeout(() => {
      console.error('Shutdown timed out; inspect any save that was still running.');
      process.exit(process.exitCode || 0);
    }, 10000);
    deadline.unref();
    io.close(() => {
      Promise.resolve(app.close()).then(() => {
        clearTimeout(deadline);
        process.exit(process.exitCode || 0);
      }).catch((error) => {
        console.error('Shutdown failed:', error.message);
        process.exit(1);
      });
    });
  };
  process.once('SIGINT', shutdown);
  process.once('SIGTERM', shutdown);
  server.on('error', (error) => {
    console.error('HTTP server failed:', error.message);
    process.exitCode = 1;
    shutdown();
  });

  server.listen(httpPort, host, () => {
    console.log(`> Ready on http://localhost:${httpPort}`);
    console.log(`> Recordings are saved to ${SAVE_ROOT}`);
    if (replaySpec) {
      console.log('REPLAY MODE: streaming recorded data');
      publishStatus({ state: 'connected', mode: 'replay', baudRate, message: 'Replaying recorded data.' });
      replay = require('./replay')(io, replaySpec, { add: emitSample });
    } else if (testMode) {
      console.log('TEST MODE: emitting random data');
      publishStatus({ state: 'connected', mode: 'simulation', baudRate, message: 'Simulated data. Not a hardware test.' });
      const startedAt = performance.now();
      let emitted = 0;
      simulator = setInterval(() => {
        const due = Math.floor((performance.now() - startedAt) * sampleRate / 1000);
        // Pace from elapsed time; bound one catch-up after a long pause.
        if (due - emitted > 2000) {
          stream.skip();
          emitted = due - 2000;
        }
        while (emitted < due) {
          const mk = () => Math.floor(Math.random() * 2047) - 1023;
          emitSample({ timestamp: Date.now(),
            ch0: { a: mk(), e: Math.abs(mk()) }, ch1: { a: mk(), e: Math.abs(mk()) },
            ch2: { a: mk(), e: Math.abs(mk()) }, ch3: { a: mk(), e: Math.abs(mk()) } });
          emitted += 1;
        }
      }, 10);
    } else {
      reader = connect(emitSample, process.env.SERIAL_PORT, {
        baudRate, expectedChannels, onStatus: publishStatus, onInvalid: () => stream.skip(),
      });
    }
  });
}

if (require.main === module) {
  main().catch((error) => { console.error(error); process.exitCode = 1; });
}
module.exports = { main, positiveInteger, createSampleStream, createNotch50 };
