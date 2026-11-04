// Replay mode for server.js: streams recorded *_adc.csv files as `adc_data`,
// in the same format readSerial.js emits from the Arduino, so the live
// prediction tab can be tested without hardware. Recordings play back to back
// with no gap, each at its own measured rate (rows / timestamp span), looping.
// Before each one starts, a `replay_truth` event names the gesture playing.
//
//   REPLAY="<file|folder|@list.txt>[;...]" node server.js
//
// @list.txt = a text file with one CSV path per line.

const fs = require("fs");
const path = require("path");

const GESTURES = ["nothing", "grasp", "flexion", "extension", "pronation",
                  "supination", "open", "left", "right"];
const GESTURE_RE = new RegExp(`_(${GESTURES.join("|")})(?:_set\\d+)?_adc\\.csv$`);

function expand(p) {
  if (p.startsWith("@")) {
    return fs.readFileSync(p.slice(1), "utf8").split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
  }
  if (fs.statSync(p).isDirectory()) {
    return fs.readdirSync(p).filter((f) => f.endsWith("_adc.csv")).sort().map((f) => path.join(p, f));
  }
  return [p];
}

function loadRecording(file) {
  const lines = fs.readFileSync(file, "utf8").trim().split(/\r?\n/);
  const header = lines[0].split(",");
  const col = (name) => header.indexOf(name);
  const pairs = [0, 1, 2, 3].map((c) => [col(`Ch${c} Act`), col(`Ch${c} Env`)]).filter(([a]) => a >= 0);
  const rows = lines.slice(1).map((l) => l.split(","));
  const t0 = Date.parse(rows[0][col("Timestamp")]);
  const t1 = Date.parse(rows[rows.length - 1][col("Timestamp")]);
  const samples = rows.map((r) => {
    const d = {};
    pairs.forEach(([a, e], c) => { d[`ch${c}`] = { a: Number(r[a]), e: Number(r[e]) }; });
    return d;
  });
  const m = path.basename(file).match(GESTURE_RE);
  return { file, gesture: m ? m[1] : null, samples, rate: samples.length / ((t1 - t0) / 1000) };
}

module.exports = function startReplay(io, spec) {
  const files = spec.split(";").map((s) => s.trim()).filter(Boolean).flatMap(expand);
  const recordings = files.map(loadRecording).filter((r) => r.gesture && isFinite(r.rate));
  if (!recordings.length) throw new Error(`REPLAY: no usable recordings in "${spec}"`);
  const meanRate = recordings.reduce((s, r) => s + r.rate, 0) / recordings.length;
  console.log(`REPLAY: ${recordings.length} recordings, mean rate ${meanRate.toFixed(0)} Hz, looping`);

  let idx = 0, sent = 0, start = 0;
  const begin = () => {
    const r = recordings[idx];
    sent = 0;
    start = Date.now();
    io.emit("replay_truth", { gesture: r.gesture, file: path.basename(r.file), index: idx });
  };
  begin();
  setInterval(() => {
    const r = recordings[idx];
    const due = Math.min(r.samples.length, Math.floor(((Date.now() - start) / 1000) * r.rate));
    for (; sent < due; sent++) io.emit("adc_data", r.samples[sent]);
    if (sent >= r.samples.length) {
      idx = (idx + 1) % recordings.length;
      begin();
    }
  }, 10);
};
