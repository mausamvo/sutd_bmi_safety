"use client";

import { useEffect, useRef, useState } from "react";
import io from "socket.io-client";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

export default function Page() {
  const motions = [
    {
      key: "nothing",
      label: "Nothing (Relaxed)",
      instructions: "Relax your hand and forearm.",
      video: "/motions/nothing.mp4",
      image: "/motions/nothing.svg",
    },
    {
      key: "grasp",
      label: "Grasp",
      instructions: "Close your hand into a firm fist.",
      video: "/motions/grasp.mp4",
      image: "/motions/grasp.svg",
    },
    {
      key: "flexion",
      label: "Flexion",
      instructions: "Bend your wrist upward.",
      video: "/motions/flexion.mp4",
      image: "/motions/flexion.svg",
    },
    {
      key: "extension",
      label: "Extension",
      instructions: "Bend your wrist downward.",
      video: "/motions/extension.mp4",
      image: "/motions/extension.svg",
    },
    {
      key: "pronation",
      label: "Pronation",
      instructions: "Rotate your forearm so palm faces down.",
      video: "/motions/pronation.mp4",
      image: "/motions/pronation.svg",
    },
    {
      key: "supination",
      label: "Supination",
      instructions: "Rotate your forearm so palm faces up.",
      video: "/motions/supination.mp4",
      image: "/motions/supination.svg",
    },
    {
      key: "open",
      label: "Open",
      instructions: "Open your fingers wide and keep them straight.",
      video: "/motions/open.mp4",
      image: "/motions/open.svg",
    },
    {
      key: "left",
      label: "Left",
      instructions: "Move in the left direction as shown.",
      video: "/motions/left.mp4",
      image: "/motions/left.svg",
    },
    {
      key: "right",
      label: "Right",
      instructions: "Move in the right direction as shown.",
      video: "/motions/right.mp4",
      image: "/motions/right.svg",
    },
  ];

  const ch0Ref = useRef(null);
  const ch1Ref = useRef(null);
  const ch2Ref = useRef(null);
  const ch3Ref = useRef(null);
  const charts = [
    { ref: ch0Ref, name: "ch0" },
    { ref: ch1Ref, name: "ch1" },
    { ref: ch2Ref, name: "ch2" },
    { ref: ch3Ref, name: "ch3" },
  ];

  const [maxChannels, setMaxChannels] = useState(4);
  const [maxY, setMaxY] = useState(500);
  const [minY, setMinY] = useState(-150);
  const socketRef = useRef();
  const dataPointsRef = useRef([]);
  const stopAutomationRef = useRef(false);
  const [action, setAction] = useState("nothing");
  const [extraFilename, setExtraFilename] = useState("");
  const [duration, setDuration] = useState(1); // controls both view and record
  const [recording, setRecording] = useState(false);
  const [countdown, setCountdown] = useState(0);
  const [paused, setPaused] = useState(false);
  const [showNotification, setShowNotification] = useState(false);
  const [notificationMessage, setNotificationMessage] = useState("");
  const [dataPoints, setDataPoints] = useState([]);
  const [activeTab, setActiveTab] = useState("manual");
  const [includeCharts, setIncludeCharts] = useState(true);

  const [setsPerMotion, setSetsPerMotion] = useState(5);
  const [cooldownSeconds, setCooldownSeconds] = useState(10);
  const [betweenMotionsCooldownSeconds, setBetweenMotionsCooldownSeconds] =
    useState(5);
  const [prepSeconds, setPrepSeconds] = useState(3);
  const [autoCaptureDuration, setAutoCaptureDuration] = useState(1);
  const [automationRunning, setAutomationRunning] = useState(false);
  const [automationPhase, setAutomationPhase] = useState("idle");
  const [automationSecondsLeft, setAutomationSecondsLeft] = useState(0);
  const [currentMotionIndex, setCurrentMotionIndex] = useState(0);
  const [currentSet, setCurrentSet] = useState(1);
  const [automationMessage, setAutomationMessage] = useState(
    "Configure settings and press Start Automated Recording."
  );
  const [lastSavedFiles, setLastSavedFiles] = useState([]);
  const [saveSubfolder, setSaveSubfolder] = useState("baseline_data_8_april");

  useEffect(() => {
    socketRef.current = io();
    const socket = socketRef.current;
    socket.on("adc_data", (d) => {
      if (paused) return;
      const entry = { timestamp: Date.now(), ...d };
      setDataPoints((prev) => {
        const cutoff = Date.now() - duration * 1000;
        const next = [entry, ...prev].filter((x) => x.timestamp >= cutoff);
        // Always append to ref so recording captures fresh data from click
        dataPointsRef.current = [...dataPointsRef.current, entry];
        return next;
      });
    });
    return () => socket.disconnect();
  }, [paused, duration]);

  useEffect(() => {
    return () => {
      stopAutomationRef.current = true;
    };
  }, []);

  const labels = dataPoints.map((d) =>
    new Date(d.timestamp).toLocaleTimeString("en-SG", {
      hour12: false,
      minute: "2-digit",
      second: "2-digit",
      fractionalSecondDigits: 3,
    })
  );

  const commonOptions = {
    animation: false,
    scales: {
      x: { title: { display: true, text: "Time" } },
      y: {
        min: minY,
        max: maxY,
        ticks: { stepSize: 5 },
        title: { display: true, text: "Value" },
      },
    },
    plugins: { legend: { position: "top" } },
  };

  const chartDataCh0 = {
    labels,
    datasets: [
      {
        label: "Channel 0 Activation",
        data: dataPoints.map((dp) => dp.ch0?.a || 0),
        borderColor: "red",
      },
      {
        label: "Channel 0 Envelope",
        data: dataPoints.map((dp) => dp.ch0?.e || 0),
        borderColor: "pink",
      },
    ],
  };

  const chartDataCh1 = {
    labels,
    datasets: [
      {
        label: "Channel 1 Activation",
        data: dataPoints.map((dp) => dp.ch1?.a || 0),
        borderColor: "green",
      },
      {
        label: "Channel 1 Envelope",
        data: dataPoints.map((dp) => dp.ch1?.e || 0),
        borderColor: "lightgreen",
      },
    ],
  };

  const chartDataCh2 = {
    labels,
    datasets: [
      {
        label: "Channel 2 Activation",
        data: dataPoints.map((dp) => dp.ch2?.a || 0),
        borderColor: "blue",
      },
      {
        label: "Channel 2 Envelope",
        data: dataPoints.map((dp) => dp.ch2?.e || 0),
        borderColor: "lightblue",
      },
    ],
  };

  const chartDataCh3 = {
    labels,
    datasets: [
      {
        label: "Channel 3 Activation",
        data: dataPoints.map((dp) => dp.ch3?.a || 0),
        borderColor: "brown",
      },
      {
        label: "Channel 3 Envelope",
        data: dataPoints.map((dp) => dp.ch3?.e || 0),
        borderColor: "black",
      },
    ],
  };

  const combinedColors = ["#d7191c", "#2ca02c", "#1f77b4", "#8c564b"];
  const chartDataCombined = {
    labels,
    datasets: Array.from({ length: maxChannels }, (_, i) => ({
      label: `Channel ${i} Activation`,
      data: dataPoints.map((dp) => dp[`ch${i}`]?.a || 0),
      borderColor: combinedColors[i] || "#333",
      pointRadius: 0,
      borderWidth: 2,
      tension: 0.2,
    })),
  };

  const createActionFilename = (action, extra) => {
    if (!extra) return action;
    return `${action}_${extra}`;
  };

  const saveAllToRepo = async (filenameSuffix, fileActionLabel) => {
    const now = Date.now();
    const data = dataPointsRef.current;

    const channelHeaders = [];
    for (let i = 0; i < maxChannels; i++) {
      channelHeaders.push(`Ch${i} Act`, `Ch${i} Env`);
    }
    channelHeaders.push("Action");

    const rows = [
      ["Timestamp", ...channelHeaders],
      ...data.map((d) => {
        const row = [new Date(d.timestamp).toISOString()];
        for (let i = 0; i < maxChannels; i++) {
          row.push(d[`ch${i}`]?.a || 0, d[`ch${i}`]?.e || 0);
        }
        row.push(fileActionLabel);
        return row;
      }),
    ];
    const csv = rows.map((r) => r.join(",")).join("\n");

    const chartImages = includeCharts
      ? charts
          .slice(0, maxChannels)
          .filter(({ ref }) => Boolean(ref.current))
          .map(({ ref, name }) => ({
            name,
            dataUrl: ref.current.toBase64Image(),
          }))
      : [];

    const response = await fetch("/api/save-capture", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        baseFilename: `${now}_${filenameSuffix}`,
        csv,
        chartImages,
        saveSubfolder,
      }),
    });

    const result = await response.json();
    if (!response.ok) {
      throw new Error(result?.error || "Failed to save capture files.");
    }

    setLastSavedFiles(result.files || []);
    return result;
  };

  const recordAndSave = async (
    durationSeconds,
    filenameSuffix,
    fileActionLabel,
    showDoneToast = true
  ) => {
    dataPointsRef.current = []; // silently reset ref to ensure recording starts fresh from click
    setRecording(true);
    let remaining = durationSeconds;
    setCountdown(remaining);
    const interval = setInterval(() => {
      remaining -= 1;
      setCountdown(remaining);
      if (remaining <= 0) clearInterval(interval);
    }, 1000);
    await new Promise((res) => setTimeout(res, durationSeconds * 1000));
    clearInterval(interval);
    const result = await saveAllToRepo(filenameSuffix, fileActionLabel);
    if (showDoneToast) {
      setNotificationMessage(
        `Saved ${result.files?.length || 0} file(s) to ${result.saveDirectory}`
      );
      setShowNotification(true);
      setTimeout(() => setShowNotification(false), 3000);
    }
    setRecording(false);
    setCountdown(0);
  };

  const waitSeconds = async (seconds, phase, messageBuilder) => {
    setAutomationPhase(phase);
    for (let i = seconds; i > 0; i -= 1) {
      if (stopAutomationRef.current) return false;
      setAutomationSecondsLeft(i);
      setAutomationMessage(messageBuilder(i));
      await new Promise((res) => setTimeout(res, 1000));
    }
    setAutomationSecondsLeft(0);
    return !stopAutomationRef.current;
  };

  const startAutomatedRecording = async () => {
    if (!saveSubfolder.trim()) {
      setAutomationMessage("Please set a save subfolder first.");
      return;
    }
    if (setsPerMotion < 1) {
      setAutomationMessage("Sets per motion must be at least 1.");
      return;
    }

    stopAutomationRef.current = false;
    setAutomationRunning(true);
    setPaused(false);
    setAutomationMessage("Automated recording will start after cooldown.");

    const cooldownOk = await waitSeconds(
      cooldownSeconds,
      "cooldown",
      (s) => `Get ready. Global cooldown: ${s}s`
    );
    if (!cooldownOk) {
      setAutomationRunning(false);
      setAutomationPhase("idle");
      setAutomationMessage("Automated recording cancelled.");
      return;
    }

    for (let motionIndex = 0; motionIndex < motions.length; motionIndex += 1) {
      const motion = motions[motionIndex];
      setCurrentMotionIndex(motionIndex);

      for (let setIndex = 1; setIndex <= setsPerMotion; setIndex += 1) {
        if (stopAutomationRef.current) {
          setAutomationRunning(false);
          setAutomationPhase("idle");
          setAutomationMessage("Automated recording cancelled.");
          return;
        }

        setCurrentSet(setIndex);

        const prepOk = await waitSeconds(
          prepSeconds,
          "prep",
          (s) =>
            `${motion.label} (set ${setIndex}/${setsPerMotion}) starts in ${s}...`
        );
        if (!prepOk) {
          setAutomationRunning(false);
          setAutomationPhase("idle");
          setAutomationMessage("Automated recording cancelled.");
          return;
        }

        setAutomationPhase("recording");
        setAutomationMessage(
          `Recording ${motion.label} (set ${setIndex}/${setsPerMotion})`
        );

        const fileActionLabel = `${motion.key}_${saveSubfolder}_set${setIndex}`;
        const filenameSuffix = `${saveSubfolder}_${motion.key}_set${setIndex}`;
        await recordAndSave(
          autoCaptureDuration,
          filenameSuffix,
          fileActionLabel,
          false
        );
      }

      const hasNextMotion = motionIndex < motions.length - 1;
      if (hasNextMotion && betweenMotionsCooldownSeconds > 0) {
        const nextMotion = motions[motionIndex + 1];
        setCurrentMotionIndex(motionIndex + 1);
        const betweenOk = await waitSeconds(
          betweenMotionsCooldownSeconds,
          "between-motions",
          (s) => `Next motion: ${nextMotion.label} in ${s}s`
        );
        if (!betweenOk) {
          setAutomationRunning(false);
          setAutomationPhase("idle");
          setAutomationMessage("Automated recording cancelled.");
          return;
        }
      }
    }

    setAutomationRunning(false);
    setAutomationPhase("done");
    setAutomationSecondsLeft(0);
    setAutomationMessage("Automated recording complete. All files saved in repo.");
    setNotificationMessage("Automated run complete. Files were saved automatically.");
    setShowNotification(true);
    setTimeout(() => setShowNotification(false), 3000);
  };

  const stopAutomatedRecording = () => {
    stopAutomationRef.current = true;
    setAutomationRunning(false);
    setAutomationPhase("idle");
    setAutomationSecondsLeft(0);
    setAutomationMessage("Stopping automated recording...");
  };

  const currentMotion = motions[currentMotionIndex] || motions[0];
  const manualFileAction = createActionFilename(action, extraFilename);

  return (
    <main style={{ width: "90%", margin: "1rem auto" }}>
      {showNotification && (
        <div
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            width: "100%",
            background: "#4caf50",
            color: "white",
            textAlign: "center",
            zIndex: 9999,
          }}
        >
          ✅ {notificationMessage || "Files saved to repo."}
        </div>
      )}

      <h2>Live ADC (last {duration}s)</h2>
      <div style={{ marginTop: 12, marginBottom: 12 }}>
        <button
          onClick={() => setActiveTab("manual")}
          disabled={recording || automationRunning}
          style={{
            marginRight: 8,
            background: activeTab === "manual" ? "#0f4c81" : "#d9d9d9",
            color: activeTab === "manual" ? "#fff" : "#111",
            border: "none",
            padding: "8px 12px",
            borderRadius: 8,
            cursor: "pointer",
          }}
        >
          Manual
        </button>
        <button
          onClick={() => setActiveTab("automated")}
          disabled={recording || automationRunning}
          style={{
            background: activeTab === "automated" ? "#0f4c81" : "#d9d9d9",
            color: activeTab === "automated" ? "#fff" : "#111",
            border: "none",
            padding: "8px 12px",
            borderRadius: 8,
            cursor: "pointer",
          }}
        >
          Automated
        </button>
      </div>

      <div style={{ marginBottom: 16, marginTop: 16 }}>
        <style>{`
          @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0; }
          }
          .recording-dot {
            display: inline-block;
            width: 12px;
            height: 12px;
            background: red;
            border-radius: 50%;
            animation: blink 1s infinite;
            margin-right: 8px;
            vertical-align: middle;
          }
        `}</style>
        <label>
          Window (s):
          <input
            type="number"
            min="1"
            max="60"
            value={duration}
            onChange={(e) => setDuration(Number(e.target.value))}
            style={{ width: 50, marginLeft: 8 }}
            disabled={recording || automationRunning}
          />
        </label>
        {activeTab === "manual" && (
          <button
            onClick={() => recordAndSave(duration, manualFileAction, manualFileAction)}
            disabled={recording || automationRunning}
            style={{ marginLeft: 16 }}
          >
            {recording ? (
              <>
                <span className="recording-dot" />
                <span style={{ color: "red" }}>{`Recording... ${countdown}s left`}</span>
              </>
            ) : (
              `Record & Save (${duration}s)`
            )}
          </button>
        )}
        <button
          onClick={() => setPaused((p) => !p)}
          style={{ marginLeft: 16 }}
          disabled={recording || automationRunning}
        >
          {paused ? "Resume" : "Pause"}
        </button>
        {activeTab === "manual" && (
          <>
            <label style={{ marginLeft: 16 }}>
              Filename:
              <select
                value={action}
                onChange={(e) => setAction(e.target.value)}
                style={{ width: 100, marginLeft: 8 }}
                disabled={recording || automationRunning}
              >
                <option value="disconnected">disconnected</option>
                <option value="nothing">nothing</option>
                <option value="supination">supination</option>
                <option value="flexion">flexion</option>
                <option value="grasp">grasp</option>
                <option value="pronation">pronation</option>
                <option value="extension">extension</option>
                <option value="open">open</option>
                <option value="left">left</option>
                <option value="right">right</option>
              </select>
            </label>
            <label style={{ marginLeft: 16 }}>
              Extra:
              <input
                value={extraFilename}
                onChange={(e) => setExtraFilename(e.target.value)}
                style={{ width: 100, marginLeft: 8 }}
                disabled={recording || automationRunning}
              />
            </label>
          </>
        )}
        <label style={{ marginLeft: 16 }}>
          Max Channels:
          <input
            type="number"
            min="1"
            max="4"
            value={maxChannels}
            onChange={(e) => setMaxChannels(Number(e.target.value))}
            style={{ width: 30, marginLeft: 8 }}
            disabled={recording || automationRunning}
          />
        </label>
        <label style={{ marginLeft: 16 }}>
          Max Y:
          <input
            type="number"
            step="10"
            value={maxY}
            onChange={(e) => setMaxY(Number(e.target.value))}
            style={{ width: 50, marginLeft: 8 }}
          />
        </label>
        <label style={{ marginLeft: 16 }}>
          Min Y:
          <input
            type="number"
            step="10"
            value={minY}
            onChange={(e) => setMinY(Number(e.target.value))}
            style={{ width: 50, marginLeft: 8 }}
          />
        </label>
      </div>

      {activeTab === "automated" && (
        <section
          style={{
            display: "grid",
            gridTemplateColumns: "2fr 1fr",
            gap: "1rem",
            marginBottom: "1.25rem",
            background: "#eef7ff",
            border: "1px solid #b8d8f2",
            borderRadius: 12,
            padding: "1rem",
          }}
        >
          <div>
            <h3>Automated Data Collection</h3>
            <p style={{ marginBottom: 12 }}>
              Sequence: cooldown - 3/2/1 prep - auto record - next motion.
            </p>
            <div style={{ display: "flex", gap: "0.8rem", flexWrap: "wrap" }}>
              <label>
                Save Subfolder:
                <input
                  value={saveSubfolder}
                  onChange={(e) => setSaveSubfolder(e.target.value)}
                  placeholder="baseline_data_8_april"
                  disabled={automationRunning || recording}
                  style={{ marginLeft: 8, width: 220 }}
                />
              </label>
            </div>
            <div
              style={{
                display: "flex",
                gap: "0.8rem",
                flexWrap: "wrap",
                marginTop: 12,
              }}
            >
              <label>
                Sets / Motion:
                <input
                  type="number"
                  min="1"
                  max="50"
                  value={setsPerMotion}
                  onChange={(e) => setSetsPerMotion(Number(e.target.value))}
                  disabled={automationRunning || recording}
                  style={{ width: 60, marginLeft: 8 }}
                />
              </label>
              <label>
                Cooldown (s):
                <input
                  type="number"
                  min="1"
                  max="60"
                  value={cooldownSeconds}
                  onChange={(e) => setCooldownSeconds(Number(e.target.value))}
                  disabled={automationRunning || recording}
                  style={{ width: 60, marginLeft: 8 }}
                />
              </label>
              <label>
                Between Motions (s):
                <input
                  type="number"
                  min="0"
                  max="60"
                  value={betweenMotionsCooldownSeconds}
                  onChange={(e) =>
                    setBetweenMotionsCooldownSeconds(Number(e.target.value))
                  }
                  disabled={automationRunning || recording}
                  style={{ width: 60, marginLeft: 8 }}
                />
              </label>
              <label>
                Prep Countdown (s):
                <input
                  type="number"
                  min="1"
                  max="10"
                  value={prepSeconds}
                  onChange={(e) => setPrepSeconds(Number(e.target.value))}
                  disabled={automationRunning || recording}
                  style={{ width: 60, marginLeft: 8 }}
                />
              </label>
              <label>
                Capture Duration (s):
                <input
                  type="number"
                  min="1"
                  max="10"
                  value={autoCaptureDuration}
                  onChange={(e) => setAutoCaptureDuration(Number(e.target.value))}
                  disabled={automationRunning || recording}
                  style={{ width: 60, marginLeft: 8 }}
                />
              </label>
            </div>

            <div style={{ marginTop: 12 }}>
              {!automationRunning ? (
                <button onClick={startAutomatedRecording} disabled={recording}>
                  Start Automated Recording
                </button>
              ) : (
                <button onClick={stopAutomatedRecording}>Stop</button>
              )}
              <label style={{ marginLeft: 16 }}>
                <input
                  type="checkbox"
                  checked={includeCharts}
                  onChange={(e) => setIncludeCharts(e.target.checked)}
                  disabled={automationRunning || recording}
                  style={{ marginRight: 6 }}
                />
                Save chart PNGs too
              </label>
            </div>

            <div
              style={{
                marginTop: 12,
                background: "#fff",
                borderRadius: 8,
                border: "1px solid #d5d5d5",
                padding: "0.8rem",
              }}
            >
              <div>
                <strong>Status:</strong> {automationPhase}
                {automationSecondsLeft > 0 && ` (${automationSecondsLeft}s)`}
              </div>
              <div>
                <strong>Current Motion:</strong> {currentMotion.label}
              </div>
              <div>
                <strong>Current Set:</strong> {currentSet}/{setsPerMotion}
              </div>
              <div style={{ marginTop: 6 }}>
                <strong>Instruction:</strong> {automationMessage}
              </div>
              <div style={{ marginTop: 6 }}>
                <strong>Do this now:</strong> {currentMotion.instructions}
              </div>
              <div style={{ marginTop: 6 }}>
                <strong>Save Folder:</strong> src/testbed/sutd_bmi_safety_data/{saveSubfolder}
              </div>
              {lastSavedFiles.length > 0 && (
                <div style={{ marginTop: 6 }}>
                  <strong>Last Saved:</strong> {lastSavedFiles.join(", ")}
                </div>
              )}
            </div>
          </div>

          <div
            style={{
              border: "1px solid #d5d5d5",
              borderRadius: 10,
              background: "#fff",
              padding: "0.8rem",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <h4 style={{ marginBottom: 8 }}>Motion Example</h4>
            {currentMotion.video ? (
              <video
                key={currentMotion.video}
                src={currentMotion.video}
                autoPlay
                muted
                loop
                playsInline
                controls={false}
                style={{ width: "100%", maxWidth: 420, height: "auto" }}
              />
            ) : (
              <img
                src={currentMotion.image}
                alt={`${currentMotion.label} example`}
                style={{ width: "100%", maxWidth: 420, height: "auto" }}
              />
            )}
            <div style={{ marginTop: 8, textAlign: "center" }}>
              {currentMotion.label}
            </div>
          </div>
        </section>
      )}

      <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
        <div>
          <h3 style={{ textAlign: "center", marginBottom: 8 }}>Quadrant View</h3>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
              gap: "1rem",
            }}
          >
            <div>
              <h4 style={{ textAlign: "center", margin: "0 0 8px" }}>Channel 0</h4>
              <Line ref={ch0Ref} data={chartDataCh0} options={commonOptions} />
            </div>
            <div>
              <h4 style={{ textAlign: "center", margin: "0 0 8px" }}>Channel 1</h4>
              <Line ref={ch1Ref} data={chartDataCh1} options={commonOptions} />
            </div>
            <div>
              <h4 style={{ textAlign: "center", margin: "0 0 8px" }}>Channel 2</h4>
              <Line ref={ch2Ref} data={chartDataCh2} options={commonOptions} />
            </div>
            <div>
              <h4 style={{ textAlign: "center", margin: "0 0 8px" }}>Channel 3</h4>
              <Line ref={ch3Ref} data={chartDataCh3} options={commonOptions} />
            </div>
          </div>
        </div>

        <div style={{ flex: "1 1 100%" }}>
          <h3 style={{ textAlign: "center" }}>Combined View (All Channels)</h3>
          <Line data={chartDataCombined} options={commonOptions} />
        </div>
      </div>
    </main>
  );
}