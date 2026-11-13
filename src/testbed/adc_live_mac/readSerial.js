// Serial reader (macOS-optimised, from sutd_bmi_safety adc-live).
// Firmware protocol: 115200 baud, one line per sample,
//   ch0_act,ch0_env,ch1_act,ch1_env,ch2_act,ch2_env,ch3_act,ch3_env\n
// Corrupt rows are rejected (never turned into zeros). The reader owns exactly
// one port and one retry timer, so an unplug (which fires both "error" and
// "close") cannot start two parallel readers.
const findArduinoPort = require('./findSerial');

function parseSample(line, expectedChannels) {
  const fields = line.trim().split(',');
  if (fields.length < 2 || fields.length > 8 || fields.length % 2) return null;
  if (expectedChannels !== undefined && fields.length !== expectedChannels * 2) return null;
  const values = fields.map((field) => {
    const value = field.trim();
    return /^-?\d+$/.test(value) ? Number(value) : NaN;
  });
  if (!values.every(Number.isSafeInteger)) return null;
  return Object.fromEntries(Array.from({ length: values.length / 2 }, (_, i) => [
    `ch${i}`, { a: values[2 * i], e: values[2 * i + 1] },
  ]));
}

// A lost newline must not grow the line buffer without limit.
function createLineDecoder(onLine, onInvalid = () => {}, maxLength = 256) {
  let pending = '';
  let discarding = false;
  return (chunk) => {
    for (const byte of Buffer.from(chunk)) {
      if (byte === 10) {
        if (!discarding && pending.trim()) onLine(pending);
        pending = '';
        discarding = false;
      } else if (!discarding) {
        pending += String.fromCharCode(byte);
        if (pending.length > maxLength) {
          pending = '';
          discarding = true;
          onInvalid('Serial row exceeded the maximum length.');
        }
      }
    }
  };
}

// onSample(sample) receives { timestamp, ch0: {a, e}, ... }. The timestamp is
// when this Mac received the row, not a device acquisition time.
function connect(onSample, portPath, options = {}) {
  const SerialPort = options.SerialPort || require('serialport').SerialPort;
  const findPort = options.findPort || findArduinoPort;
  const baudRate = options.baudRate || 115200;
  const expectedChannels = options.expectedChannels ?? 4;
  const retryDelay = options.retryDelay ?? 2000;
  const silenceTimeout = options.silenceTimeout ?? 5000;
  const onStatus = options.onStatus || (() => {});
  const onInvalid = options.onInvalid || (() => {});
  let stopped = false;
  let active = null;
  let retryTimer = null;
  let generation = 0;
  let lastState;

  function status(state, message, path = portPath) {
    if (stopped && state !== 'stopped') return;
    lastState = state;
    onStatus({ state, port: path || null, baudRate, ...(message ? { message } : {}) });
  }

  function retry(error, token) {
    if (stopped || token !== generation || retryTimer) return;
    generation += 1; // invalidate both close/error callbacks before closing
    const previous = active;
    active = null;
    status('reconnecting', error?.message || 'Serial connection closed.');
    const schedule = () => {
      if (stopped || retryTimer) return;
      retryTimer = setTimeout(() => {
        retryTimer = null;
        void open();
      }, retryDelay);
    };
    if (previous?.port.isOpen) {
      previous.port.removeListener('data', previous.onData);
      previous.port.close(schedule);
    } else {
      schedule();
    }
  }

  async function open() {
    if (stopped) return;
    const token = ++generation;
    status('connecting');
    try {
      // Discover on startup AND on reconnect: a replug can change the path.
      const path = portPath || await findPort();
      if (stopped || token !== generation) return;
      const port = new SerialPort({ path, baudRate, autoOpen: false });
      const onData = createLineDecoder((line) => {
        if (stopped || token !== generation) return;
        active.lastDataAt = Date.now();
        const sample = parseSample(line, expectedChannels);
        if (!sample) {
          onInvalid('Malformed serial row was rejected.');
          status('invalid_data', 'Malformed serial row was rejected.', path);
          return;
        }
        if (lastState !== 'connected') status('connected', undefined, path);
        onSample({ timestamp: Date.now(), ...sample });
      }, (reason) => {
        if (!stopped && token === generation) {
          onInvalid(reason);
          status('invalid_data', reason, path);
        }
      });
      active = { port, onData, token, lastDataAt: Date.now() };
      port.on('data', onData);
      port.on('error', (error) => retry(error, token));
      port.on('close', () => retry(new Error('Serial device disconnected.'), token));
      port.open((error) => {
        if (stopped || token !== generation) {
          if (port.isOpen) port.close(() => {});
          return;
        }
        if (error) retry(error, token);
        // "connected" is reported once valid rows arrive, not when USB opens.
      });
    } catch (error) {
      retry(error, token);
    }
  }

  const watchdog = setInterval(() => {
    if (active && Date.now() - active.lastDataAt > silenceTimeout) {
      retry(new Error('No serial samples received. Reopening the receiver.'), active.token);
    }
  }, Math.max(5, Math.min(1000, silenceTimeout / 2)));
  void open();
  return {
    stop() {
      if (stopped) return;
      stopped = true;
      generation += 1;
      clearInterval(watchdog);
      clearTimeout(retryTimer);
      retryTimer = null;
      if (active) {
        active.port.removeListener('data', active.onData);
        if (active.port.isOpen) active.port.close(() => {});
      }
      active = null;
      status('stopped');
    },
  };
}

module.exports = connect;
module.exports.parseSample = parseSample;
module.exports.createLineDecoder = createLineDecoder;
