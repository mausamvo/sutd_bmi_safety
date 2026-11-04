// server.js
const { createServer } = require('http');
const next = require('next');
const socketio = require('socket.io');
const connect = require('./readSerial');
const findESP32Port = require('./findSerial');

const dev = process.env.NODE_ENV !== 'production';
const app = next({ dev });
const handle = app.getRequestHandler();

const TEST_MODE = process.env.TEST_MODE === 'true';
const REPLAY = process.env.REPLAY; // recorded CSVs instead of serial, see replay.js
const SAMPLE_RATE = +process.env.SAMPLE_RATE || 100;
const HTTP_PORT = +process.env.HTTP_PORT || 3000;

// Events passed between the browser's Live tab and the Python prediction
// sidecar (new_ml_code: python -m src.live serve), which is a socket client.
const LIVE_EVENTS = ['live_cmd', 'live_prediction', 'live_status'];

const delay = (ms) => new Promise((res) => setTimeout(res, ms));

// Resolve the serial port: use SERIAL_PORT if set, otherwise auto-detect,
// retrying forever so a missing board never crashes or blocks the server.
async function resolvePort() {
  if (process.env.SERIAL_PORT) return process.env.SERIAL_PORT;
  for (;;) {
    try {
      return await findESP32Port();
    } catch {
      console.log('Arduino not found; retrying in 2s...');
      await delay(2000);
    }
  }
}

// Runs independently of the HTTP server so the UI loads with or without hardware.
async function startSerial(io) {
  const portPath = await resolvePort();
  console.log('Reading from serial port', portPath);
  // On a permanent disconnect, re-resolve (re-detects a changed COM number).
  connect(io, portPath, () => startSerial(io));
}

app.prepare().then(() => {
  const server = createServer((req, res) => handle(req, res));
  const io = socketio(server, { cors: { origin: '*' } });

  io.on('connection', (socket) => {
    console.log('Client connected:', socket.id);
    for (const event of LIVE_EVENTS) {
      socket.on(event, (msg) => socket.broadcast.emit(event, msg));
    }
  });

  // Serve the UI first, so it is reachable regardless of serial state.
  server.listen(HTTP_PORT, () => {
    console.log(`> Ready on http://localhost:${HTTP_PORT}`);
  });

  if (REPLAY) {
    console.log('REPLAY MODE emitting recorded data');
    require('./replay')(io, REPLAY);
  } else if (TEST_MODE) {
    console.log('TEST MODE emitting random data');
    setInterval(() => {
      const mk = () => Math.floor(Math.random() * 2047) - 1023;
      io.emit('adc_data', {
        ch0: { a: mk(), e: mk() },
        ch1: { a: mk(), e: mk() },
      });
    }, 1000 / SAMPLE_RATE);
  } else {
    startSerial(io);
  }
});
