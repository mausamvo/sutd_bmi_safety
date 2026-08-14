# adc-live

Live sEMG viewer + capture tool. A Next.js app with a custom Node server that reads
ADC samples from an ESP32/Arduino over serial, streams them to the browser via
socket.io, plots them with Chart.js, and saves recordings as CSV (plus optional
chart PNGs) into `../sutd_bmi_safety_data/<subfolder>`.

## Run

```bash
pnpm install
pnpm dev            # reads from serial port (auto-detects the board)
pnpm start:test     # no hardware needed; emits random data
```

Then open http://localhost:3000.

### Environment variables

| Var           | Default                | Purpose                                                |
|---------------|------------------------|--------------------------------------------------------|
| `SERIAL_PORT` | auto-detected          | Serial device path (e.g. `COM5`, `/dev/ttyUSB0`).      |
| `SAMPLE_RATE` | `100`                  | Only used by `TEST_MODE` to pace the fake emitter.     |
| `HTTP_PORT`   | `3000`                 | HTTP + socket.io port.                                 |
| `TEST_MODE`   | `false`                | If `true`, skip serial and emit random values.         |

Auto-detection matches Arduino / CH340 / CP210x / usbserial / Prolific / ACM0 on
`SerialPort.list()`. If your board doesn't match, set `SERIAL_PORT` explicitly.

## Serial protocol

Baud `115200`, newline-delimited. Each line is a comma-separated list of integers,
in `(activation, envelope)` pairs per channel — up to 4 channels:

```
ch0_act, ch0_env, ch1_act, ch1_env, ch2_act, ch2_env, ch3_act, ch3_env
```

Example traffic (2 channels):

```
290,128,6,127
286,132,6,133
288,131,6,132
```

Non-numeric fields become `0`. The server emits an `adc_data` socket.io event
shaped as `{ ch0: { a, e }, ch1: { a, e }, ... }`.

## UI

Two modes, toggled with tabs:

- **Manual** — pick an action label (grasp, flexion, etc.), optional filename
  suffix, capture window in seconds, and hit *Record & Save*.
- **Automated** — walks through all motions (nothing, grasp, flexion, extension,
  pronation, supination, open, left, right) for N sets each, with configurable
  cooldown / prep / capture / between-motion durations. Shows a demo video for
  each motion.

Controls (both modes):
- Window (s), Max Channels (1-4), Min/Max Y, Pause/Resume.
- *Save chart PNGs too* (automated mode) also writes per-channel chart images
  alongside the CSV.

## Output

`POST /api/save-capture` writes to `../sutd_bmi_safety_data/<saveSubfolder>/`
(resolved from `process.cwd()`, so run from this directory). Filenames:

- `<timestamp>_<suffix>_adc.csv`
- `<timestamp>_<suffix>_ch<0-3>.png` (if chart PNGs enabled)

CSV columns: `Timestamp, Ch0 Act, Ch0 Env, ..., Ch<N-1> Act, Ch<N-1> Env, Action`.

## Scripts

| Script       | What it does                                            |
|--------------|---------------------------------------------------------|
| `dev`        | `node server.js` (dev Next + serial).                   |
| `start`      | Same, `NODE_ENV=development` explicit.                  |
| `start:test` | `TEST_MODE=true` — random data, no serial.              |
| `build`      | `next build`.                                           |
| `lint`       | ESLint (`lint:fix` to autofix).                         |
