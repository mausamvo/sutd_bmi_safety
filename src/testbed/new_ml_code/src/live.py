"""
Live gesture prediction with the per-subject classical model, plus the
calibration step that trains that model from one person's baseline recordings.

The live path reuses the offline pipeline's own steps --
resampling to config.TARGET_SAMPLE_RATE_HZ, a config.WINDOW_SIZE window and
window_to_td_features -- so a window scored live is built exactly like a
window the model was trained on.

  Train / calibrate (one or more folders of *_adc.csv baseline recordings):
    python -m src.live train --name sri ../sutd_bmi_safety_data/sri_baseline-29Sept26 ...
    python -m src.live train --name sri --eval <folders>   # also reports CV accuracy

  Live use, next to the adc-live web app (its "Live prediction" tab):
    python -m src.live serve [--url http://localhost:3000] [--model sri]

Models are saved to new_ml_code/live_models/<name>.joblib (git-ignored).
"""
import argparse
import os
import re
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict

from . import config
from .dataset import _raw_windows_from_dataframe, _resample_to_target
from .features import window_to_td_features

MODEL_DIR = config.PROJECT_ROOT / "live_models"

# Fixed hyperparameters, close to what the grouped-CV search picked for the
# three Sept/Oct subjects (ExtraTrees every time), so calibration takes
# seconds instead of minutes. Pass search=True to train() to run the search.
DEFAULT_PARAMS = {"n_estimators": 500, "max_depth": 25, "min_samples_leaf": 4,
                  "max_features": 0.5, "class_weight": "balanced"}

_GESTURE_RE = re.compile(r"_(" + "|".join(config.GESTURES) + r")(?:_set\d+)?_adc\.csv$")

# Resampling zero-pads the ends of the buffer, which distorts the last few
# output samples; the live window stops this many samples short of "now".
_EDGE_SAMPLES = 10
# Corrections from the live app are weighted so a few seconds of labelled
# windows can move a model trained on thousands.
CORRECTION_WEIGHT = 5.0


# ---------------------------------------------------------------------------
# Calibration / training
# ---------------------------------------------------------------------------
def recording_files(folders):
    """Every *_adc.csv under the given folders whose name carries a gesture."""
    files = []
    for folder in folders:
        for path in sorted(Path(folder).glob("*_adc.csv")):
            m = _GESTURE_RE.search(path.name)
            if m:
                files.append((path, m.group(1)))
    return files


def features_from_files(files):
    X, y, groups = [], [], []
    gesture_to_idx = {g: i for i, g in enumerate(config.GESTURES)}
    for path, gesture in files:
        for w in _raw_windows_from_dataframe(pd.read_csv(path)):
            X.append(window_to_td_features(w))
            y.append(gesture_to_idx[gesture])
            groups.append(str(path))
    if not X:
        raise ValueError("No windows found -- check the folders hold *_<gesture>[_setN]_adc.csv files")
    return np.stack(X).astype(np.float32), np.array(y), np.array(groups)


def _new_model(params):
    return ExtraTreesClassifier(random_state=config.RF_RANDOM_STATE, n_jobs=-1, **params)


def evaluate(X, y, groups, params=DEFAULT_PARAMS, n_splits=3):
    """Grouped-by-recording CV: how well a model from these recordings
    generalises to recordings it did not see. Window and recording accuracy."""
    n_splits = min(n_splits, len(set(groups)))
    pred = cross_val_predict(_new_model(params), X, y, groups=groups,
                             cv=GroupKFold(n_splits=n_splits))
    df = pd.DataFrame({"g": groups, "y": y, "p": pred})
    per_file = df.groupby("g").agg(y=("y", "first"), p=("p", lambda s: s.mode().iloc[0]))
    return {"window_acc": float((pred == y).mean()),
            "recording_acc": float((per_file.y == per_file.p).mean()),
            "n_recordings": int(len(per_file))}


def train(name, folders, params=None, search=False, extra=None, log=print):
    """Fit a model on the recordings in `folders` and save it as `name`.
    `extra` = (X, y, weights) of additional labelled windows (corrections)."""
    files = recording_files(folders)
    log(f"{len(files)} recordings from {len(folders)} folder(s)")
    X, y, groups = features_from_files(files)
    params = dict(params or DEFAULT_PARAMS)
    if search:
        from .classical import fit_best_classical_model
        _, est, results = fit_best_classical_model(X, y, groups_train=groups)
        params = results["ExtraTrees"]["best_params"]
    weights = np.ones(len(y))
    X_fit, y_fit = X, y
    if extra is not None and len(extra[1]):
        X_fit = np.concatenate([X, extra[0]])
        y_fit = np.concatenate([y, extra[1]])
        weights = np.concatenate([weights, extra[2]])
    model = _new_model(params).fit(X_fit, y_fit, sample_weight=weights)
    model.set_params(n_jobs=1)  # saved for live use, which scores one window at a time
    bundle = {
        "name": name, "model": model, "gestures": list(config.GESTURES),
        "folders": [str(f) for f in folders], "params": params,
        "n_recordings": len(files), "n_windows": int(len(y)),
        "corrections": extra if extra is not None else
        (np.empty((0, X.shape[1]), np.float32), np.empty(0, int), np.empty(0)),
        "created": datetime.now().isoformat(timespec="seconds"),
        "pipeline": {"window": config.WINDOW_SIZE, "stride": config.WINDOW_STRIDE,
                     "target_rate": config.TARGET_SAMPLE_RATE_HZ,
                     "td_features": config.TD_FEATURES},
    }
    save(bundle)
    log(f"saved {model_path(name)}  ({len(y)} windows, params {params})")
    return bundle, (X, y, groups)


def model_path(name):
    return MODEL_DIR / f"{name}.joblib"


def save(bundle):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path(bundle["name"]))


def load(name):
    bundle = joblib.load(model_path(name))
    expected = {"window": config.WINDOW_SIZE, "target_rate": config.TARGET_SAMPLE_RATE_HZ,
                "td_features": config.TD_FEATURES}
    got = {k: bundle["pipeline"][k] for k in expected}
    if got != expected:
        raise ValueError(f"{name} was trained with {got}, current config is {expected}")
    return bundle


def available_models():
    return sorted(p.stem for p in MODEL_DIR.glob("*.joblib")) if MODEL_DIR.exists() else []


# ---------------------------------------------------------------------------
# Live prediction
# ---------------------------------------------------------------------------
class LivePredictor:
    """Feed one 8-signal sample at a time with push(); call predict() as often
    as you like. The sample rate is measured from arrival times, because the
    model needs every window resampled to config.TARGET_SAMPLE_RATE_HZ."""

    def __init__(self, bundle=None, buffer_s=2.0, rate_window_s=5.0, smooth_s=0.5):
        self._lock = threading.Lock()
        self.buffer_s, self.rate_window_s, self.smooth_s = buffer_s, rate_window_s, smooth_s
        self.samples = deque()   # (t, np.ndarray(8,))
        self.recent = deque()    # (t, proba, features) for smoothing + corrections
        self.bundle = self.model = None
        if bundle is not None:
            self.set_bundle(bundle)

    def set_bundle(self, bundle):
        with self._lock:
            self.bundle = bundle
            self.model = bundle["model"]
            # One window at a time: thread start-up costs more than it saves
            # (~75 ms per prediction with n_jobs=-1 vs ~18 ms single-threaded).
            self.model.set_params(n_jobs=1)
            # model.classes_ are gesture indices; map probabilities onto all gestures
            self._cols = np.asarray(self.model.classes_, dtype=int)
            self.recent.clear()

    def push(self, sample, t=None):
        t = time.monotonic() if t is None else t
        with self._lock:
            self.samples.append((t, np.asarray(sample, dtype=np.float32)))
            keep = max(self.buffer_s, self.rate_window_s)
            while self.samples and t - self.samples[0][0] > keep:
                self.samples.popleft()

    def rate_hz(self):
        if len(self.samples) < 20:
            return None
        t_now = self.samples[-1][0]
        ts = [t for t, _ in self.samples if t_now - t <= self.rate_window_s]
        span = ts[-1] - ts[0]
        return (len(ts) - 1) / span if span >= 1.0 else None

    def predict(self):
        """Returns None until ~1 s of signal has arrived, else a dict with the
        smoothed gesture, its confidence, the raw (unsmoothed) gesture,
        per-gesture probabilities and the measured rate."""
        with self._lock:
            rate = self.rate_hz()
            if rate is None or self.model is None:
                return None
            t_now = self.samples[-1][0]
            chunk = [s for t, s in self.samples if t_now - t <= self.buffer_s]
            sig = np.stack(chunk, axis=1)                       # (8, T) at native rate
            sig = _resample_to_target(sig, rate)  # (8, T') at 120 Hz
            if sig.shape[1] < config.WINDOW_SIZE + _EDGE_SAMPLES:
                return None
            window = sig[:, -(config.WINDOW_SIZE + _EDGE_SAMPLES):-_EDGE_SAMPLES]
            feats = window_to_td_features(window)
            proba = np.zeros(len(config.GESTURES))
            proba[self._cols] = self.model.predict_proba(feats[None])[0]

            self.recent.append((t_now, proba, feats))
            while self.recent and t_now - self.recent[0][0] > max(self.smooth_s, 1.0):
                self.recent.popleft()
            smoothed = np.mean([p for t, p, _ in self.recent if t_now - t <= self.smooth_s], axis=0)
        best = int(smoothed.argmax())
        return {"gesture": config.GESTURES[best], "confidence": float(smoothed[best]),
                "raw_gesture": config.GESTURES[int(proba.argmax())],
                "probs": dict(zip(config.GESTURES, smoothed.round(3).tolist())),
                "rate_hz": rate}

    def recent_features(self, seconds=1.0):
        """Feature vectors of the windows scored in the last `seconds` -- what a
        correction labels."""
        with self._lock:
            if not self.recent:
                return np.empty((0, config.NUM_TD_FEATURES), np.float32)
            t_now = self.recent[-1][0]
            return np.stack([f for t, _, f in self.recent if t_now - t <= seconds])


def corrected_bundle(bundle, corrections, log=print):
    """Retrain `bundle` on its original folders plus its earlier corrections
    plus `corrections` -- a list of (feature windows, true gesture). Saved
    under the same name; returns the new bundle."""
    Xc, yc, wc = bundle["corrections"]
    for X_new, gesture in corrections:
        idx = config.GESTURES.index(gesture)
        Xc = np.concatenate([Xc, X_new])
        yc = np.concatenate([yc, np.full(len(X_new), idx)])
        wc = np.concatenate([wc, np.full(len(X_new), CORRECTION_WEIGHT)])
    new, _ = train(bundle["name"], bundle["folders"], params=bundle["params"],
                   extra=(Xc, yc, wc), log=log)
    return new


# ---------------------------------------------------------------------------
# Sidecar for the adc-live web app
# ---------------------------------------------------------------------------
# Where adc-live's capture tab saves recordings (app/api/save-capture/route.js).
RAW_DATA_ROOT = Path(os.environ.get("SEMG_RAW_DATA", config.PROJECT_ROOT.parent / "sutd_bmi_safety_data"))
PREDICT_EVERY_S = 0.1


def recording_folders(root=RAW_DATA_ROOT):
    """Subfolders of the capture root holding gesture recordings, as
    {"name": path relative to root, "recordings": count}."""
    out = []
    for d in sorted(p for p in [root, *root.rglob("*")] if p.is_dir()):
        n = len(recording_files([d]))
        if n:
            out.append({"name": d.relative_to(root).as_posix() or ".", "recordings": n})
    return out


def serve(url, model_name=None):
    """Connect to adc-live's Socket.IO server as a client: score `adc_data`
    live, emit `live_prediction`, and act on the browser's `live_cmd`
    messages (load / train / correct). server.js relays both directions."""
    import socketio

    sio = socketio.Client(reconnection=True, reconnection_attempts=0, reconnection_delay=2)
    predictor = LivePredictor()
    state = {"message": "", "training": False, "pending": [], "bad_samples": 0,
             "last_sample": 0.0, "folders": recording_folders()}

    def model_info():
        b = predictor.bundle
        if b is None:
            return None
        folders = []
        for f in b["folders"]:
            try:
                folders.append(Path(f).resolve().relative_to(RAW_DATA_ROOT.resolve()).as_posix())
            except ValueError:
                folders.append(str(f))
        return {"name": b["name"], "recordings": b["n_recordings"], "created": b["created"],
                "folders": folders, "corrections": int(len(b["corrections"][1]))}

    def status(message=None):
        if message is not None:
            state["message"] = message
            print(f"[live] {message}")
        sio.emit("live_status", {
            "model": model_info(), "models": available_models(), "folders": state["folders"],
            "training": state["training"], "message": state["message"],
            "pending_corrections": sum(len(x) for x, _ in state["pending"]),
            "streaming": time.monotonic() - state["last_sample"] < 2.0,
            "rate_hz": predictor.rate_hz(), "bad_samples": state["bad_samples"],
            "data_root": str(RAW_DATA_ROOT),
        })

    def load_model(name):
        predictor.set_bundle(load(name))
        state["pending"] = []
        status(f"Loaded model '{name}'")

    def in_background(job, label):
        if state["training"]:
            status("Already training -- wait for it to finish")
            return
        state["training"] = True
        status(label)

        def run():
            try:
                job()
            except Exception as exc:  # report to the browser rather than kill the sidecar
                status(f"Failed: {exc}")
            finally:
                state["training"] = False
                status()
        threading.Thread(target=run, daemon=True).start()

    @sio.on("adc_data")
    def on_adc(d):
        try:
            sample = [d[f"ch{c}"][k] for c in range(4) for k in ("a", "e")]
        except (KeyError, TypeError):
            state["bad_samples"] += 1  # e.g. TEST_MODE emits only 2 channels
            return
        predictor.push(sample)
        state["last_sample"] = time.monotonic()

    @sio.on("live_cmd")
    def on_cmd(cmd):
        kind = (cmd or {}).get("type")
        if kind == "status":
            state["folders"] = recording_folders()
            status()
        elif kind == "load":
            try:
                load_model(cmd["name"])
            except Exception as exc:
                status(f"Could not load '{cmd.get('name')}': {exc}")
        elif kind == "train":
            name = cmd.get("name", "").strip()
            folders = [RAW_DATA_ROOT / f for f in cmd.get("folders", [])]
            if not name or not folders:
                status("Give the model a name and pick at least one folder")
                return

            def job():
                bundle, _ = train(name, folders, log=status)
                predictor.set_bundle(bundle)
                state["pending"] = []
                state["folders"] = recording_folders()
                status(f"Trained and loaded '{name}' ({bundle['n_recordings']} recordings)")
            in_background(job, f"Training '{name}' on {len(folders)} folder(s)...")
        elif kind == "correct":
            gesture = cmd.get("gesture")
            X = predictor.recent_features(1.0)
            if gesture not in config.GESTURES or not len(X):
                status("Nothing to correct yet")
                return
            state["pending"].append((X, gesture))
            status(f"Marked the last {len(X)} windows as '{gesture}'")
        elif kind == "apply_corrections":
            if predictor.bundle is None or not state["pending"]:
                status("No corrections to apply")
                return
            pending, state["pending"] = state["pending"], []

            def job():
                predictor.set_bundle(corrected_bundle(predictor.bundle, pending, log=status))
                status("Retrained with corrections")
            in_background(job, "Retraining with corrections...")
        elif kind == "discard_corrections":
            state["pending"] = []
            status("Discarded pending corrections")

    @sio.event
    def connect():
        print(f"[live] connected to {url}")
        status()

    def predict_loop():
        while True:
            t = time.monotonic()
            if sio.connected:
                out = predictor.predict()
                if out is not None:
                    sio.emit("live_prediction", out)
            time.sleep(max(0.0, PREDICT_EVERY_S - (time.monotonic() - t)))

    def status_loop():
        while True:
            time.sleep(1.0)
            if sio.connected:
                status()

    if model_name is None and available_models():
        model_name = max(available_models(), key=lambda n: model_path(n).stat().st_mtime)
    if model_name:
        predictor.set_bundle(load(model_name))
        state["message"] = f"Loaded model '{model_name}'"
    threading.Thread(target=predict_loop, daemon=True).start()
    threading.Thread(target=status_loop, daemon=True).start()
    while True:
        try:
            sio.connect(url, transports=["websocket"])
            sio.wait()
        except socketio.exceptions.ConnectionError as exc:
            print(f"[live] cannot reach {url} ({exc}), retrying in 2 s")
            time.sleep(2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="train/calibrate a model from baseline folders")
    t.add_argument("--name", required=True)
    t.add_argument("--eval", action="store_true", help="also report grouped-CV accuracy")
    t.add_argument("--search", action="store_true", help="hyperparameter search (minutes)")
    t.add_argument("folders", nargs="+")
    sub.add_parser("list", help="list saved models")
    s = sub.add_parser("serve", help="live prediction sidecar for the adc-live web app")
    s.add_argument("--url", default=os.environ.get("SOCKET_URL", "http://localhost:3000"))
    s.add_argument("--model", help="model to load at start (default: most recent)")
    args = ap.parse_args()
    if args.cmd == "serve":
        serve(args.url, args.model)
        return
    if args.cmd == "list":
        for n in available_models():
            b = joblib.load(model_path(n))
            print(f"{n:20s} {b['n_recordings']:4d} recordings  {b['created']}")
        return
    t0 = time.time()
    _, (X, y, groups) = train(args.name, args.folders, search=args.search)
    if args.eval:
        r = evaluate(X, y, groups)
        print(f"grouped CV over {r['n_recordings']} recordings: window {r['window_acc']:.3f}, "
              f"recording {r['recording_acc']:.3f}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
