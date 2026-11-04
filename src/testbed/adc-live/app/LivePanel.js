"use client";

// "Live prediction" tab. Talks to the Python sidecar
// (new_ml_code: python -m src.live serve) through server.js:
//   live_prediction  sidecar -> browser, ~10 per second
//   live_status      sidecar -> browser, every second and after each command
//   live_cmd         browser -> sidecar: status | load | train | correct |
//                    apply_corrections | discard_corrections
// In REPLAY mode, server.js also sends replay_truth before each recording,
// which this panel uses to score the predictions against the known gesture.

import { useEffect, useRef, useState } from "react";
import io from "socket.io-client";

// Predictions this soon after a replayed recording starts still see the
// previous recording in their ~0.8 s window, so they are not scored.
const SETTLE_MS = 1200;

const card = {
  background: "#ffffff",
  border: "1px solid #d6dde4",
  borderRadius: 12,
  padding: "1rem",
};
const buttonStyle = {
  border: "1px solid #9fb3c8",
  background: "#f4f7fa",
  borderRadius: 8,
  padding: "6px 10px",
  cursor: "pointer",
  color: "#111111",
};
const primary = { ...buttonStyle, background: "#0f4c81", color: "#ffffff", border: "none" };

export default function LivePanel({ motions }) {
  const socketRef = useRef(null);
  const [status, setStatus] = useState(null);
  const [statusAt, setStatusAt] = useState(0);
  const [prediction, setPrediction] = useState(null);
  const [now, setNow] = useState(Date.now());
  const [loadName, setLoadName] = useState("");
  const [trainName, setTrainName] = useState("");
  const [trainFolders, setTrainFolders] = useState([]);

  // Replay scoring
  const truthRef = useRef(null); // { gesture, start, preds: [] }
  const [truth, setTruth] = useState(null);
  const [score, setScore] = useState({ windows: 0, windowsOk: 0, recordings: 0, recordingsOk: 0 });

  const labelOf = (key) => motions.find((m) => m.key === key)?.label || key;

  useEffect(() => {
    const socket = io();
    socketRef.current = socket;
    socket.on("connect", () => socket.emit("live_cmd", { type: "status" }));
    socket.on("live_status", (s) => {
      setStatus(s);
      setStatusAt(Date.now());
    });
    socket.on("live_prediction", (p) => {
      setPrediction(p);
      const t = truthRef.current;
      if (t && Date.now() - t.start >= SETTLE_MS) {
        t.preds.push(p.raw_gesture);
        setScore((s) => ({
          ...s,
          windows: s.windows + 1,
          windowsOk: s.windowsOk + (p.raw_gesture === t.gesture ? 1 : 0),
        }));
      }
    });
    socket.on("replay_truth", (r) => {
      const prev = truthRef.current;
      if (prev && prev.preds.length) {
        const counts = {};
        prev.preds.forEach((g) => (counts[g] = (counts[g] || 0) + 1));
        const majority = Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
        setScore((s) => ({
          ...s,
          recordings: s.recordings + 1,
          recordingsOk: s.recordingsOk + (majority === prev.gesture ? 1 : 0),
        }));
      }
      truthRef.current = { gesture: r.gesture, start: Date.now(), preds: [] };
      setTruth(r);
    });
    const tick = setInterval(() => setNow(Date.now()), 500);
    return () => {
      clearInterval(tick);
      socket.disconnect();
    };
  }, []);

  useEffect(() => {
    if (status?.model?.name && !loadName) setLoadName(status.model.name);
  }, [status, loadName]);

  const send = (cmd) => socketRef.current?.emit("live_cmd", cmd);
  const sidecarUp = now - statusAt < 3000;
  const busy = !sidecarUp || status?.training;
  const fresh = prediction && sidecarUp && status?.streaming;
  const pct = (x) => `${Math.round(x * 100)}%`;

  const toggleFolder = (name) =>
    setTrainFolders((f) => (f.includes(name) ? f.filter((x) => x !== name) : [...f, name]));

  return (
    <section
      style={{
        display: "grid",
        gridTemplateColumns: "minmax(0, 2fr) minmax(0, 1fr)",
        gap: "1rem",
        marginBottom: "1.25rem",
        background: "#eef7ff",
        border: "1px solid #b8d8f2",
        borderRadius: 12,
        padding: "1rem",
        color: "#111111",
      }}
    >
      <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
        <div style={card}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
            <h3 style={{ margin: 0 }}>Live prediction</h3>
            <span style={{ color: "#4a5568", fontSize: 14 }}>
              {!sidecarUp
                ? "Sidecar not connected"
                : !status?.model
                  ? "No model loaded"
                  : !status.streaming
                    ? "Waiting for sensor data"
                    : `${status.rate_hz ? status.rate_hz.toFixed(0) : "…"} Hz · model ${status.model.name}`}
            </span>
          </div>
          <div
            style={{
              fontSize: 56,
              fontWeight: 700,
              textAlign: "center",
              padding: "1.2rem 0 0.4rem",
              color: fresh ? "#0f4c81" : "#9aa5b1",
            }}
          >
            {fresh ? labelOf(prediction.gesture) : "—"}
          </div>
          <div style={{ textAlign: "center", color: "#4a5568", marginBottom: 12 }}>
            {fresh
              ? `confidence ${pct(prediction.confidence)} · this window: ${labelOf(prediction.raw_gesture)}`
              : "prediction appears about 1 s after data starts"}
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            {motions.map((m) => {
              const p = fresh ? prediction.probs[m.key] || 0 : 0;
              return (
                <div key={m.key} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ width: 150, fontSize: 14 }}>{m.label}</span>
                  <div style={{ flex: 1, background: "#e6ebf0", borderRadius: 4, height: 14 }}>
                    <div
                      style={{
                        width: pct(p),
                        height: 14,
                        borderRadius: 4,
                        background: fresh && m.key === prediction.gesture ? "#0f4c81" : "#7aa7cf",
                      }}
                    />
                  </div>
                  <span style={{ width: 44, textAlign: "right", fontSize: 14 }}>{pct(p)}</span>
                </div>
              );
            })}
          </div>
        </div>

        {truth && (
          <div style={card}>
            <h3 style={{ marginTop: 0 }}>Replay check</h3>
            <p style={{ margin: "0 0 8px" }}>
              Now playing: <b>{labelOf(truth.gesture)}</b>{" "}
              <span style={{ color: "#4a5568", fontSize: 13 }}>({truth.file})</span>
            </p>
            <p style={{ margin: 0 }}>
              Windows correct: <b>{score.windows ? pct(score.windowsOk / score.windows) : "—"}</b>{" "}
              ({score.windowsOk}/{score.windows}) · Recordings correct (majority vote):{" "}
              <b>{score.recordings ? pct(score.recordingsOk / score.recordings) : "—"}</b> (
              {score.recordingsOk}/{score.recordings})
            </p>
            <button
              style={{ ...buttonStyle, marginTop: 8 }}
              onClick={() => setScore({ windows: 0, windowsOk: 0, recordings: 0, recordingsOk: 0 })}
            >
              Reset score
            </button>
          </div>
        )}

        <div style={card}>
          <h3 style={{ marginTop: 0 }}>Correct the model</h3>
          <p style={{ margin: "0 0 8px", color: "#4a5568" }}>
            While holding a gesture, press what you are actually doing. This labels the last second
            of windows; apply to retrain.
          </p>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
            {motions.map((m) => (
              <button
                key={m.key}
                style={buttonStyle}
                disabled={busy || !status?.model}
                onClick={() => send({ type: "correct", gesture: m.key })}
              >
                {m.label}
              </button>
            ))}
          </div>
          <div style={{ marginTop: 10, display: "flex", gap: 8, alignItems: "center" }}>
            <button
              style={primary}
              disabled={busy || !status?.pending_corrections}
              onClick={() => send({ type: "apply_corrections" })}
            >
              Apply {status?.pending_corrections || 0} corrected windows
            </button>
            <button
              style={buttonStyle}
              disabled={busy || !status?.pending_corrections}
              onClick={() => send({ type: "discard_corrections" })}
            >
              Discard
            </button>
          </div>
        </div>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
        <div style={card}>
          <h3 style={{ marginTop: 0 }}>Model</h3>
          {status?.model ? (
            <p style={{ margin: "0 0 8px", fontSize: 14 }}>
              <b>{status.model.name}</b>: {status.model.recordings} recordings
              {status.model.corrections ? ` + ${status.model.corrections} corrected windows` : ""}
              <br />
              <span style={{ color: "#4a5568" }}>
                trained {status.model.created} on {status.model.folders.join(", ")}
              </span>
            </p>
          ) : (
            <p style={{ margin: "0 0 8px" }}>No model loaded. Calibrate one below.</p>
          )}
          <div style={{ display: "flex", gap: 8 }}>
            <select
              value={loadName}
              onChange={(e) => setLoadName(e.target.value)}
              style={{ flex: 1 }}
              disabled={busy || !status?.models?.length}
            >
              {(status?.models || []).map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
            <button
              style={buttonStyle}
              disabled={busy || !loadName}
              onClick={() => send({ type: "load", name: loadName })}
            >
              Load
            </button>
          </div>
        </div>

        <div style={card}>
          <h3 style={{ marginTop: 0 }}>Calibrate</h3>
          <p style={{ margin: "0 0 8px", color: "#4a5568", fontSize: 14 }}>
            Record this person&apos;s baselines in the Automated tab, then train on those folders.
            Calibrate on the same day, without moving the sensors.
          </p>
          <label style={{ display: "block", marginBottom: 8 }}>
            Model name:
            <input
              value={trainName}
              onChange={(e) => setTrainName(e.target.value)}
              placeholder="e.g. sri"
              style={{ marginLeft: 8, width: 140 }}
              disabled={busy}
            />
          </label>
          <div
            style={{
              maxHeight: 220,
              overflowY: "auto",
              border: "1px solid #d6dde4",
              borderRadius: 8,
              padding: 6,
              fontSize: 14,
            }}
          >
            {(status?.folders || []).map((f) => (
              <label key={f.name} style={{ display: "block" }}>
                <input
                  type="checkbox"
                  checked={trainFolders.includes(f.name)}
                  onChange={() => toggleFolder(f.name)}
                  disabled={busy}
                />{" "}
                {f.name} <span style={{ color: "#4a5568" }}>({f.recordings})</span>
              </label>
            ))}
            {!status?.folders?.length && <span>No recording folders found.</span>}
          </div>
          <div style={{ marginTop: 8, display: "flex", gap: 8 }}>
            <button
              style={primary}
              disabled={busy || !trainName.trim() || !trainFolders.length}
              onClick={() => send({ type: "train", name: trainName.trim(), folders: trainFolders })}
            >
              {status?.training ? "Training…" : `Train on ${trainFolders.length} folder(s)`}
            </button>
            <button style={buttonStyle} disabled={!sidecarUp} onClick={() => send({ type: "status" })}>
              Refresh folders
            </button>
          </div>
        </div>

        <div style={{ ...card, fontSize: 13, color: "#4a5568" }}>
          {status?.message || (sidecarUp ? "Ready" : "Start the sidecar: python -m src.live serve")}
          {status?.bad_samples ? (
            <div>
              {status.bad_samples} samples skipped: the model needs all 4 channels (TEST_MODE sends 2).
            </div>
          ) : null}
        </div>
      </div>
    </section>
  );
}
