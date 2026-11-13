# adc_live_mac

The `1009891-live-inference` branch of `sutd_bmi_safety` (`src/testbed/adc-live`),
including its Manual / Automated / Live prediction tabs, with the macOS-optimised
data handling from `~/Downloads/sutd_bmi_safety` underneath. The page looks and
behaves like the branch; the changes are in how samples are moved, plotted and saved.

## Run

```bash
npm ci              # once
npm run dev         # real board (auto-detected)
npm run start:test  # no board: random 4-channel data
```

Open http://localhost:3000. If the board is not found, set its port explicitly:
`SERIAL_PORT=/dev/cu.usbserial-XXXX npm run dev`
(list ports: `node -e "require('serialport').SerialPort.list().then(console.log)"`).
Close Arduino Serial Monitor first: only one program can hold the port.

Recordings are saved to `new/data/recordings/<Save Subfolder>/` (set `SAVE_ROOT`
to change). The CSV format and file names are the branch's:
`<time>_<subfolder>_<motion>_set<n>_adc.csv`, Action `<motion>_<subfolder>_set<n>`.

## Automated tab: choose motions

Tick the motions to record in the **Motions** row (all are ticked by default).
The run records only those, in the usual order.

## 50 Hz filter button (all tabs)

**OFF (default):** every value is saved exactly as the board sends it, mains hum
included. Use OFF for all attack and baseline recordings.

**ON:** the server removes a narrow band around 50 Hz from the activation values
(Act) before plotting, saving and live prediction. Its sample rate is measured from
the incoming rows. Things to know:
- Env cannot be cleaned: the board computes it before the data reaches the Mac.
- The 100 Hz and 150 Hz harmonics are not removed.
- Files are named `<time>_notch50_...` so they are never mixed up with unfiltered
  files. Keep them in a separate Save Subfolder: the analysis code reads them like
  any other recording.
- The button is locked during a recording, and a capture never mixes filtered and
  unfiltered samples.

The board's own firmware filter is unchanged. It is set for 60 Hz at 500 samples/s,
so at the real ~335 rows/s its notch sits near 40 Hz, not 50 Hz.

## Live prediction tab

Runs a per-person model in a Python helper that uses `../new_ml_code` for its features.
Start it in a second terminal, in this folder, while the app is running:

```bash
python live.py serve
```

Then record the person's baselines in the Automated tab, and in **Live prediction →
Calibrate** tick those folders and press Train. Models are saved in `live_models/`.
Needs `python-socketio` and `websocket-client` (both already installed in this Mac's
Anaconda Python) plus new_ml_code's own packages.

Test without the board: `REPLAY="../../data/recordings/<folder>" npm run dev`
streams recorded CSVs and the tab scores its predictions ("Replay check").

## What changed from the branch (and why)

- Samples come from the server in acknowledged batches with a sequence number and the
  time this Mac received each row. The charts redraw at most 20 times a second (at most
  600 points per line, spikes kept) instead of once per sample, so the page doesn't lag.
- Pausing the plot or changing the window no longer reconnects, and recording keeps
  working while the plot is paused (before, a paused plot recorded nothing).
- A capture is saved only if every sample arrived. Lost samples, an unplugged board or
  a lost connection cancel that capture. The automated run then repeats that set after
  the prep countdown (up to 3 tries) instead of hanging.
- Stop cancels at once and never saves a half capture. A failed save no longer leaves
  the page stuck on "Recording".
- Corrupt serial rows are rejected (before, they became zeros). An unplugged board is
  re-detected. A red note next to "samples/s" appears only when there is a problem.
- Security: the server only listens on this Mac and accepts this page only. Saves
  refuse odd folder names and never overwrite. The live helper accepts only safe model
  names and folders inside the recordings folder. Next.js is updated to 15.5.29
  (15.3.1 had critical known vulnerabilities).
- The Mac version's firmware is NOT included. It keeps the same 60 Hz notch, but it
  changes the sample timing and serial output, and it has never been uploaded to a
  board, so recordings would no longer match earlier sessions. Keep the current firmware.
