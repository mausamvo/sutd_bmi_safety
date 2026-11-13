"use client";

import { useEffect, useRef, useState } from "react";
import io from "socket.io-client";
import { Line } from "react-chartjs-2";
import LivePanel from "./LivePanel";
import {
  SampleBuffer,
  CaptureBuffer,
  decimateForDisplay,
  waitFor,
  csvCell,
  DISPLAY_INTERVAL_MS,
} from "./live-buffer";
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

// One formatter for every time label (same text as toLocaleTimeString).
const timeLabel = new Intl.DateTimeFormat("en-SG", {
  hour12: false,
  minute: "2-digit",
  second: "2-digit",
  fractionalSecondDigits: 3,
});
// How long to wait for the first sample after a capture window: it proves
// every sample inside the window has arrived.
const CAPTURE_GRACE_MS = 300;
// A failed automated set (lost samples, board unplugged, save error) is
// recorded again after the prep countdown, up to this many attempts.
const MAX_SET_ATTEMPTS = 3;

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
  const [socket, setSocket] = useState(null);
  const bufferRef = useRef(null); // bounded display history
  const capturesRef = useRef(new Set()); // captures in progress (lossless)
  const dirtyRef = useRef(false); // new samples since the last redraw
  const pausedRef = useRef(false);
  const durationRef = useRef(1);
  const abortRef = useRef(null); // cancels the current recording / run
  const runningRef = useRef(false); // a recording or run (incl. its save) is in progress
  const noticeTimerRef = useRef(null);
  const sampleCountRef = useRef(0);
  const stopAutomationRef = useRef(false);
  const pauseAutomationRef = useRef(false);
  const [automationPaused, setAutomationPaused] = useState(false);
  const [liveHz, setLiveHz] = useState(0);
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
  const [linkStatus, setLinkStatus] = useState(null);
  const [socketConnected, setSocketConnected] = useState(null);
  const [saveRootLabel, setSaveRootLabel] = useState("new/data/recordings");
  const [notificationError, setNotificationError] = useState(false);
  // Which motions the automated run records (all ticked by default).
  const [selectedMotionKeys, setSelectedMotionKeys] = useState(() =>
    motions.map((m) => m.key)
  );
  // 50 Hz mains-hum filter, applied by the server; off = exactly what the board sends.
  const [notch50, setNotch50] = useState(false);

  useEffect(() => {
    // One socket for the whole page (the Live tab uses it too). Changing the
    // window or pausing the plot no longer reconnects.
    if (!bufferRef.current) bufferRef.current = new SampleBuffer();
    const s = io({ reconnectionDelay: 1000 });
    setSocket(s);
    let reconnectTimer = null;
    const failCaptures = (message) => {
      for (const capture of capturesRef.current) capture.fail(message);
    };
    // Samples arrive in batches of ~20 ms. Every sample goes to the display
    // buffer and to any capture in progress, even while the plot is paused.
    s.on("adc_batch", (samples, ack) => {
      if (typeof ack === "function") ack(); // acknowledge first, independent of drawing
      if (!Array.isArray(samples)) return;
      for (const sample of samples) {
        if (!sample || !Number.isFinite(sample.timestamp)) continue;
        bufferRef.current.push(sample);
        for (const capture of capturesRef.current) capture.push(sample);
      }
      sampleCountRef.current += samples.length; // counted per second for the live rate readout
      dirtyRef.current = true;
    });
    s.on("serial_status", (status) => {
      setLinkStatus(status);
      if (status?.state === "error" || status?.state === "reconnecting") {
        failCaptures(`${status.message || "Board disconnected."} Capture was not saved; please retry.`);
      }
    });
    s.on("filter_status", (filter) => {
      if (typeof filter?.notch50 === "boolean") setNotch50(filter.notch50);
    });
    s.on("app_config", (config) => {
      if (typeof config?.saveRootLabel === "string") setSaveRootLabel(config.saveRootLabel);
    });
    s.on("connect", () => setSocketConnected(true));
    s.on("disconnect", (reason) => {
      setSocketConnected(false);
      failCaptures("Lost connection to the app server. Capture was not saved; please retry.");
      // After a server-side disconnect Socket.IO does not reconnect by itself.
      if (reason === "io server disconnect") {
        if (reconnectTimer) clearTimeout(reconnectTimer);
        reconnectTimer = setTimeout(() => s.connect(), 1000);
      }
    });
    return () => {
      if (reconnectTimer) clearTimeout(reconnectTimer);
      s.disconnect();
    };
  }, []);

  useEffect(() => {
    pausedRef.current = paused;
    durationRef.current = duration;
    dirtyRef.current = true;
  }, [paused, duration]);

  useEffect(() => {
    // Redraw the charts at most 20 times a second, not once per sample.
    const id = setInterval(() => {
      if (!dirtyRef.current || pausedRef.current || document.hidden) return;
      const latest = bufferRef.current?.latest();
      if (!latest) return;
      dirtyRef.current = false;
      const recent = bufferRef.current.recent(latest.timestamp - durationRef.current * 1000);
      setDataPoints(decimateForDisplay(recent).reverse()); // newest first, as before
    }, DISPLAY_INTERVAL_MS);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    const id = setInterval(() => {
      setLiveHz(sampleCountRef.current);
      sampleCountRef.current = 0;
    }, 1000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    return () => {
      stopAutomationRef.current = true;
      abortRef.current?.abort();
      clearTimeout(noticeTimerRef.current);
    };
  }, []);

  const labels = dataPoints.map((d) => timeLabel.format(d.timestamp));

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

  const showNotice = (message, isError = false) => {
    setNotificationMessage(message);
    setNotificationError(isError);
    setShowNotification(true);
    clearTimeout(noticeTimerRef.current);
    noticeTimerRef.current = setTimeout(() => setShowNotification(false), isError ? 8000 : 3000);
  };

  // samples: every raw sample of the capture window (never the plotted subset).
  const saveAllToRepo = async (samples, channels, filenameSuffix, fileActionLabel) => {
    const now = Date.now();
    // Files recorded with the 50 Hz filter ON are named <time>_notch50_..., so
    // they can never be mistaken for unfiltered recordings.
    const filtered = samples[0]?.notch50 === true;

    const channelHeaders = [];
    for (let i = 0; i < channels; i++) {
      channelHeaders.push(`Ch${i} Act`, `Ch${i} Env`);
    }
    channelHeaders.push("Action");

    const actionCell = csvCell(fileActionLabel); // quoted only if it holds a comma
    const rows = [
      ["Timestamp", ...channelHeaders],
      ...samples.map((d) => {
        const row = [new Date(d.timestamp).toISOString()];
        for (let i = 0; i < channels; i++) {
          row.push(d[`ch${i}`].a, d[`ch${i}`].e);
        }
        row.push(actionCell);
        return row;
      }),
    ];
    const csv = rows.map((r) => r.join(",")).join("\n");

    const chartImages = includeCharts
      ? charts
          .slice(0, channels)
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
        baseFilename: filtered ? `${now}_notch50_${filenameSuffix}` : `${now}_${filenameSuffix}`,
        csv,
        chartImages,
        saveSubfolder,
      }),
    });

    const result = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(result?.error || "Failed to save capture files.");
    }

    setLastSavedFiles(result.files || []);
    return result;
  };

  // Captures every sample from the click for durationSeconds, then saves.
  // Throws (and saves nothing) if samples were lost, the board or server
  // disconnected, or the save failed; throws AbortError when cancelled.
  const recordAndSave = async (
    durationSeconds,
    filenameSuffix,
    fileActionLabel,
    showDoneToast = true,
    signal
  ) => {
    const channels = Math.min(4, Math.max(1, Math.floor(maxChannels) || 1));
    const capture = new CaptureBuffer(Date.now(), durationSeconds * 1000);
    capturesRef.current.add(capture);
    setRecording(true);
    let remaining = durationSeconds;
    setCountdown(remaining);
    const interval = setInterval(() => {
      remaining -= 1;
      setCountdown(remaining);
      if (remaining <= 0) clearInterval(interval);
    }, 1000);
    try {
      await waitFor(durationSeconds * 1000, signal);
      const graceEnd = Date.now() + CAPTURE_GRACE_MS;
      while (!capture.complete && !capture.error && Date.now() < graceEnd) {
        await waitFor(20, signal);
      }
      const samples = capture.finish(channels);
      if (samples.some((x) => x.notch50 !== samples[0].notch50)) {
        throw new Error("The 50 Hz filter was switched during the capture. Capture was not saved; please retry.");
      }
      capturesRef.current.delete(capture);
      const result = await saveAllToRepo(samples, channels, filenameSuffix, fileActionLabel);
      if (showDoneToast) {
        showNotice(
          `Saved ${result.files?.length || 0} file(s) to ${result.saveDirectory}` +
            (samples[0]?.notch50 ? " (50 Hz filter ON)" : "")
        );
      }
      return result;
    } finally {
      clearInterval(interval);
      capturesRef.current.delete(capture);
      setRecording(false);
      setCountdown(0);
    }
  };

  const recordManual = async () => {
    if (runningRef.current) return;
    runningRef.current = true;
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      await recordAndSave(duration, manualFileAction, manualFileAction, true, controller.signal);
    } catch (error) {
      if (error?.name !== "AbortError") showNotice(error?.message || "Recording failed.", true);
    } finally {
      runningRef.current = false;
      if (abortRef.current === controller) abortRef.current = null;
    }
  };

  const waitSeconds = async (seconds, phase, messageBuilder) => {
    setAutomationPhase(phase);
    try {
      for (let i = seconds; i > 0; i -= 1) {
        if (stopAutomationRef.current) return false;
        setAutomationSecondsLeft(i);
        setAutomationMessage(messageBuilder(i));
        await waitFor(1000, abortRef.current?.signal); // Stop ends it at once
      }
    } catch (error) {
      if (error?.name === "AbortError") return false;
      throw error;
    }
    setAutomationSecondsLeft(0);
    return !stopAutomationRef.current;
  };

  // Blocks the automation loop while paused. Only called between captures, so a
  // capture in progress always finishes before the pause takes effect.
  const waitWhilePaused = async () => {
    if (!pauseAutomationRef.current) return;
    setAutomationPhase("paused");
    setAutomationSecondsLeft(0);
    setAutomationMessage("Paused. Press Resume to continue with the next capture.");
    while (pauseAutomationRef.current && !stopAutomationRef.current) {
      await new Promise((res) => setTimeout(res, 200));
    }
  };

  const togglePauseAutomation = () => {
    const next = !pauseAutomationRef.current;
    pauseAutomationRef.current = next;
    setAutomationPaused(next);
  };

  const startAutomatedRecording = async () => {
    if (runningRef.current) return; // a stopped run is still finishing its save
    if (!saveSubfolder.trim()) {
      setAutomationMessage("Please set a save subfolder first.");
      return;
    }
    if (setsPerMotion < 1) {
      setAutomationMessage("Sets per motion must be at least 1.");
      return;
    }
    // Ticked motions, in the usual order.
    const runMotions = motions.filter((m) => selectedMotionKeys.includes(m.key));
    if (!runMotions.length) {
      setAutomationMessage("Tick at least one motion to record.");
      return;
    }

    runningRef.current = true;
    const controller = new AbortController();
    abortRef.current = controller;
    const { signal } = controller;
    stopAutomationRef.current = false;
    pauseAutomationRef.current = false;
    setAutomationPaused(false);
    setAutomationRunning(true);
    setPaused(false);
    setCurrentMotionIndex(motions.indexOf(runMotions[0]));
    setAutomationMessage("Automated recording will start after cooldown.");

    // Returns true when every set was saved, false when Stop was pressed.
    const runAll = async () => {
      const cooldownOk = await waitSeconds(
        cooldownSeconds,
        "cooldown",
        (s) => `Get ready. Global cooldown: ${s}s`
      );
      if (!cooldownOk) return false;

      for (let runIndex = 0; runIndex < runMotions.length; runIndex += 1) {
        const motion = runMotions[runIndex];
        setCurrentMotionIndex(motions.indexOf(motion));

        for (let setIndex = 1; setIndex <= setsPerMotion; setIndex += 1) {
          if (stopAutomationRef.current) return false;

          setCurrentSet(setIndex);

          // Hold here if paused (between captures), then continue with this set.
          await waitWhilePaused();
          if (stopAutomationRef.current) return false;

          const prepMessage = (s) =>
            `${motion.label} (set ${setIndex}/${setsPerMotion}) starts in ${s}...`;
          const prepOk = await waitSeconds(prepSeconds, "prep", prepMessage);
          if (!prepOk) return false;

          // If paused during prep, hold now — before the capture, not inside it.
          if (pauseAutomationRef.current) {
            await waitWhilePaused();
            if (stopAutomationRef.current) return false;
            // Re-run prep so the subject gets the countdown again after resuming.
            const prepOk2 = await waitSeconds(prepSeconds, "prep", prepMessage);
            if (!prepOk2) return false;
          }

          const fileActionLabel = `${motion.key}_${saveSubfolder}_set${setIndex}`;
          const filenameSuffix = `${saveSubfolder}_${motion.key}_set${setIndex}`;
          for (let attempt = 1; ; attempt += 1) {
            setAutomationPhase("recording");
            setAutomationMessage(
              `Recording ${motion.label} (set ${setIndex}/${setsPerMotion})`
            );
            try {
              await recordAndSave(
                autoCaptureDuration,
                filenameSuffix,
                fileActionLabel,
                false,
                signal
              );
              break;
            } catch (error) {
              if (error?.name === "AbortError" || stopAutomationRef.current) return false;
              if (attempt >= MAX_SET_ATTEMPTS) throw error;
              // Nothing was saved for this set: count down and record it again.
              const retryOk = await waitSeconds(
                prepSeconds,
                "prep",
                (s) =>
                  `${error?.message || "Capture failed."} Repeating ${motion.label} (set ${setIndex}/${setsPerMotion}) in ${s}...`
              );
              if (!retryOk) return false;
            }
          }
        }

        // Hold here if paused before moving to the next motion.
        await waitWhilePaused();
        if (stopAutomationRef.current) return false;

        const hasNextMotion = runIndex < runMotions.length - 1;
        if (hasNextMotion && betweenMotionsCooldownSeconds > 0) {
          const nextMotion = runMotions[runIndex + 1];
          setCurrentMotionIndex(motions.indexOf(nextMotion));
          const betweenOk = await waitSeconds(
            betweenMotionsCooldownSeconds,
            "between-motions",
            (s) => `Next motion: ${nextMotion.label} in ${s}s`
          );
          if (!betweenOk) return false;
        }
      }
      return true;
    };

    try {
      const finished = await runAll();
      setAutomationRunning(false);
      setAutomationSecondsLeft(0);
      if (finished) {
        setAutomationPhase("done");
        setAutomationMessage("Automated recording complete. All files saved in repo.");
        showNotice("Automated run complete. Files were saved automatically.");
      } else {
        setAutomationPhase("idle");
        setAutomationMessage("Automated recording cancelled.");
      }
    } catch (error) {
      setAutomationRunning(false);
      setAutomationPhase("idle");
      setAutomationSecondsLeft(0);
      const message = `Automated recording stopped: ${error?.message || "unknown error"}`;
      setAutomationMessage(message);
      showNotice(message, true);
    } finally {
      pauseAutomationRef.current = false;
      setAutomationPaused(false);
      runningRef.current = false;
      if (abortRef.current === controller) abortRef.current = null;
    }
  };

  const stopAutomatedRecording = () => {
    stopAutomationRef.current = true;
    pauseAutomationRef.current = false;
    abortRef.current?.abort(); // ends the current wait or capture; nothing half-done is saved
    setAutomationPaused(false);
    setAutomationSecondsLeft(0);
    setAutomationMessage("Stopping automated recording...");
  };

  const currentMotion = motions[currentMotionIndex] || motions[0];
  const manualFileAction = createActionFilename(action, extraFilename);

  // Subject-facing label + color for the big status banner.
  const phaseDisplay = {
    idle: { label: "Idle", color: "#6b7280" },
    cooldown: { label: "Get Ready", color: "#f59e0b" },
    prep: { label: "Get Ready", color: "#f59e0b" },
    recording: { label: "Recording", color: "#dc2626" },
    "between-motions": { label: "Next Motion", color: "#2563eb" },
    paused: { label: "Paused", color: "#9333ea" },
    done: { label: "Done", color: "#16a34a" },
  }[automationPhase] || { label: automationPhase, color: "#6b7280" };

  // Shown next to the samples/s readout only when something is wrong.
  const linkProblem =
    socketConnected === false
      ? "Not connected to the app server (is npm run dev still running?)"
      : !linkStatus || linkStatus.state === "connected"
        ? null
        : linkStatus.state === "connecting"
          ? "Waiting for the board..."
          : linkStatus.message || `Board ${linkStatus.state}`;

  return (
    <main style={{ width: "90%", margin: "1rem auto" }}>
      {showNotification && (
        <div
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            width: "100%",
            background: notificationError ? "#dc2626" : "#4caf50",
            color: "white",
            textAlign: "center",
            zIndex: 9999,
          }}
        >
          {notificationError ? "⚠️" : "✅"} {notificationMessage || "Files saved to repo."}
        </div>
      )}

      <h2>
        Live ADC (last {duration}s){" "}
        <span style={{ fontSize: "0.6em", fontWeight: "normal", color: "#555" }}>
          — {liveHz} samples/s
          {linkStatus?.mode && socketConnected ? ` (${linkStatus.mode} data)` : ""}
        </span>
        {linkProblem && (
          <span style={{ fontSize: "0.6em", fontWeight: "normal", color: "#dc2626", marginLeft: 8 }}>
            · {linkProblem}
          </span>
        )}
      </h2>
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
            color: "#111111",
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
        <button
          onClick={() => setActiveTab("live")}
          disabled={recording || automationRunning}
          style={{
            marginLeft: 8,
            background: activeTab === "live" ? "#0f4c81" : "#d9d9d9",
            color: activeTab === "live" ? "#fff" : "#111",
            border: "none",
            padding: "8px 12px",
            borderRadius: 8,
            cursor: "pointer",
          }}
        >
          Live prediction
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
            onClick={recordManual}
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
        <button
          onClick={() => socket?.emit("set_filter", { notch50: !notch50 })}
          style={{
            marginLeft: 16,
            ...(notch50
              ? { background: "#f59e0b", color: "#fff", border: "none", fontWeight: 600 }
              : {}),
          }}
          disabled={recording || automationRunning || !socket}
          title="Removes 50 Hz mains hum from the activation signal (Act). Off = data exactly as the board sends it."
        >
          {notch50 ? "50 Hz Filter: ON" : "50 Hz Filter: OFF"}
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
            color: "#111111",
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
                alignItems: "center",
                marginTop: 12,
              }}
            >
              <span>Motions:</span>
              {motions.map((m) => (
                <label key={m.key}>
                  <input
                    type="checkbox"
                    checked={selectedMotionKeys.includes(m.key)}
                    onChange={(e) =>
                      setSelectedMotionKeys((keys) =>
                        e.target.checked
                          ? [...keys, m.key]
                          : keys.filter((k) => k !== m.key)
                      )
                    }
                    disabled={automationRunning || recording}
                    style={{ marginRight: 4 }}
                  />
                  {m.label}
                </label>
              ))}
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
                <>
                  <button
                    onClick={togglePauseAutomation}
                    style={{
                      marginRight: 8,
                      background: automationPaused ? "#16a34a" : "#f59e0b",
                      color: "#fff",
                      border: "none",
                      padding: "8px 16px",
                      borderRadius: 8,
                      fontWeight: 600,
                      cursor: "pointer",
                    }}
                  >
                    {automationPaused ? "Resume" : "Pause"}
                  </button>
                  <button onClick={stopAutomatedRecording}>Stop</button>
                </>
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
                borderRadius: 12,
                border: `4px solid ${phaseDisplay.color}`,
                padding: "1.5rem",
                textAlign: "center",
              }}
            >
              {/* Big subject-facing status: phase + countdown */}
              <div
                style={{
                  fontSize: "3rem",
                  fontWeight: 800,
                  lineHeight: 1.1,
                  color: phaseDisplay.color,
                  textTransform: "uppercase",
                  letterSpacing: "0.02em",
                }}
              >
                {phaseDisplay.label}
                {automationSecondsLeft > 0 && (
                  <span style={{ marginLeft: 16 }}>{automationSecondsLeft}s</span>
                )}
              </div>

              {/* Current motion */}
              <div
                style={{
                  fontSize: "4.5rem",
                  fontWeight: 900,
                  lineHeight: 1.05,
                  margin: "0.5rem 0",
                  color: "#111",
                }}
              >
                {currentMotion.label}
              </div>

              {/* What to do right now */}
              <div style={{ fontSize: "1.75rem", color: "#333", marginBottom: 8 }}>
                {currentMotion.instructions}
              </div>

              <div style={{ fontSize: "1.5rem", color: "#555" }}>
                Set {currentSet} / {setsPerMotion}
              </div>

              {/* Operator details, kept small */}
              <div
                style={{
                  marginTop: 16,
                  paddingTop: 12,
                  borderTop: "1px solid #e5e5e5",
                  fontSize: "0.85rem",
                  color: "#888",
                  textAlign: "left",
                }}
              >
                <div>{automationMessage}</div>
                <div>
                  Save Folder: {saveRootLabel}/{saveSubfolder}
                </div>
                {lastSavedFiles.length > 0 && (
                  <div>Last Saved: {lastSavedFiles.join(", ")}</div>
                )}
              </div>
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

      {activeTab === "live" && <LivePanel motions={motions} socket={socket} />}

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