// Replay mode for server.js: streams recorded *_adc.csv files in place of the
// board, so the Live prediction tab can be tested without hardware.
// Recordings play back to back, each at its own measured rate
// (rows / timestamp span), looping. Before each one starts, a `replay_truth`
// event names the gesture playing.
//
//   REPLAY="<file|folder|@list.txt>[;...]" npm run dev
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
  const pairs = [0, 1, 2, 3].map((c) => [col(`Ch${c} Act`), col(`Ch${c} Env`)]).filter(([a, e]) => a >= 0 && e >= 0);
  const tCol = col("Timestamp");
  const rows = lines.slice(1).map((l) => l.split(","));
  const samples = [];
  for (const r of rows) {
    const d = {};
    let ok = true;
    pairs.forEach(([a, e], c) => {
      const va = Number(r[a]);
      const ve = Number(r[e]);
      if (!Number.isFinite(va) || !Number.isFinite(ve)) ok = false;
      d[`ch${c}`] = { a: va, e: ve };
    });
    if (ok) samples.push(d); // skip broken rows rather than send NaN
  }
  const t0 = Date.parse(rows[0]?.[tCol]);
  const t1 = Date.parse(rows[rows.length - 1]?.[tCol]);
  const m = path.basename(file).match(GESTURE_RE);
  return { file, gesture: m ? m[1] : null, samples, rate: samples.length / ((t1 - t0) / 1000) };
}

// stream.add(sample) = the server's sample path (filter, sequence numbers, batches).
module.exports = function startReplay(io, spec, stream) {
  const files = spec.split(";").map((s) => s.trim()).filter(Boolean).flatMap(expand);
  const recordings = files.map(loadRecording)
    .filter((r) => r.gesture && r.samples.length && Number.isFinite(r.rate) && r.rate > 0);
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
  const timer = setInterval(() => {
    const r = recordings[idx];
    const due = Math.min(r.samples.length, Math.floor(((Date.now() - start) / 1000) * r.rate));
    for (; sent < due; sent++) stream.add({ timestamp: Date.now(), ...r.samples[sent] });
    if (sent >= r.samples.length) {
      idx = (idx + 1) % recordings.length;
      begin();
    }
  }, 10);
  return { stop: () => clearInterval(timer) };
};
