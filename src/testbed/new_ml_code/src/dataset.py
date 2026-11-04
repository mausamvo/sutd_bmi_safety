"""
Turns a file-level manifest (from data.py) into windowed tensors ready for
EMGNetFusion: a CWT scalogram tensor (CNN branch) and a classical
time-domain feature vector (TD branch) per window. Windows never cross a
recording-file boundary, so the file-level split from data.py is preserved
at the window level too.

Results are cached to outputs/cache/, keyed by split name + the
windowing/feature config, so re-running the pipeline after a code change
to the model/training loop doesn't recompute every window from scratch.
"""
import hashlib
import json
from math import gcd
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from . import config
from .data import load_recording
from .features import window_to_cwt_tensor, window_to_td_features

CACHE_DIR = config.OUTPUT_DIR / "cache"

# Bumped whenever the windowing/feature CODE changes in a way not already
# captured by a config.py constant, so a stale cache is never silently reused.
# CWT (the wavelet transform, ~15-20 minutes for the full dataset) and TD
# (hand-crafted features, seconds) are cached and hashed SEPARATELY -- most
# iteration happens on TD features/classifier hyperparameters, and there is
# no reason to recompute the expensive CWT scalograms every time a TD
# feature is added or tweaked.
_WINDOW_CODE_VERSION = 1
_CWT_CODE_VERSION = 1
_TD_CODE_VERSION = 2  # v2: added frequency-domain features (see features.py)


def _windowing_payload() -> dict:
    return {
        "window_code_version": _WINDOW_CODE_VERSION,
        "subject": config.SUBJECT,
        "channels": config.CHANNEL_COLUMNS,
        "gestures": config.GESTURES,
        "window": config.WINDOW_SIZE,
        "stride": config.WINDOW_STRIDE,
        "normalize_windows": config.NORMALIZE_WINDOWS,
        "target_rate": config.TARGET_SAMPLE_RATE_HZ,
    }


def _cwt_hash() -> str:
    payload = json.dumps({
        **_windowing_payload(),
        "cwt_code_version": _CWT_CODE_VERSION,
        "scales": config.CWT_NUM_SCALES,
        "wavelet": config.CWT_WAVELET,
        "downsample": config.SPECTROGRAM_DOWNSAMPLE,
    }, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


def _td_hash() -> str:
    payload = json.dumps({
        **_windowing_payload(),
        "td_code_version": _TD_CODE_VERSION,
        "td_features": config.TD_FEATURES,
        "wamp_threshold": config.WAMP_THRESHOLD,
        "sample_rate_hz": config.EFFECTIVE_SAMPLE_RATE_HZ,
    }, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


def _normalize_window(window: np.ndarray) -> np.ndarray:
    mean = window.mean(axis=1, keepdims=True)
    std = window.std(axis=1, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (window - mean) / std


def measured_rate_hz(csv_df: pd.DataFrame) -> float:
    """Rows per second for one recording, from the span of its timestamps.

    Individual timestamps are batch-stamped by the browser and so are not
    reliable sample-by-sample, but the first-to-last span is, which is all
    this needs.
    """
    ts = pd.to_datetime(csv_df["Timestamp"])
    dur = (ts.iloc[-1] - ts.iloc[0]).total_seconds()
    if dur <= 0:
        return float("nan")
    return len(csv_df) / dur


def _resample_to_target(signal: np.ndarray, rate_hz: float) -> np.ndarray:
    """
    signal: (num_channels, T) at rate_hz -> (num_channels, T') at
    config.TARGET_SAMPLE_RATE_HZ.

    Uses polyphase resampling, which low-pass filters before decimating so
    that content above the new Nyquist limit is removed rather than folding
    back as false low-frequency content. Only ever downsamples in practice
    (target is the slowest setup's rate); a file already at the target is
    returned untouched.
    """
    target = config.TARGET_SAMPLE_RATE_HZ
    if not np.isfinite(rate_hz) or abs(rate_hz - target) / target <= config.RESAMPLE_TOLERANCE:
        return signal
    up, down = int(round(target)), int(round(rate_hz))
    g = gcd(up, down)
    return resample_poly(signal, up // g, down // g, axis=1).astype(np.float32)


def _raw_windows_from_dataframe(csv_df: pd.DataFrame) -> np.ndarray:
    """csv_df: one recording's rows. -> (num_windows, num_channels, window_size)."""
    signal = csv_df[config.CHANNEL_COLUMNS].to_numpy(dtype=np.float32).T  # (C, T)
    signal = _resample_to_target(signal, measured_rate_hz(csv_df))
    n_samples = signal.shape[1]
    windows = []
    start = 0
    while start + config.WINDOW_SIZE <= n_samples:
        w = signal[:, start:start + config.WINDOW_SIZE]
        if config.NORMALIZE_WINDOWS:
            w = _normalize_window(w)
        windows.append(w)
        start += config.WINDOW_STRIDE
    if not windows:
        return np.empty((0, config.NUM_CHANNELS, config.WINDOW_SIZE), dtype=np.float32)
    return np.stack(windows, axis=0)


def _windows_and_labels(manifest: pd.DataFrame):
    """Yields (window, label_idx, meta_row) for every window in every file in
    manifest, in a fixed order -- both the CWT and TD builders below iterate
    this identically, so X_cwt[i]/X_td[i]/y[i]/window_manifest.iloc[i] always
    refer to the same window."""
    gesture_to_idx = {g: i for i, g in enumerate(config.GESTURES)}
    for _, rec in manifest.iterrows():
        csv_df = load_recording(Path(rec["path"]))
        windows = _raw_windows_from_dataframe(csv_df)
        label_idx = gesture_to_idx[rec["gesture"]]
        for w in windows:
            yield w, label_idx, {
                "path": rec["path"], "session": rec["session"],
                "condition": rec["condition"], "gesture": rec["gesture"],
                "set_num": rec["set_num"],
            }


def _build_cwt(manifest: pd.DataFrame):
    all_cwt, all_y, rows = [], [], []
    for w, label_idx, meta in _windows_and_labels(manifest):
        all_cwt.append(window_to_cwt_tensor(w))
        all_y.append(label_idx)
        rows.append(meta)
    X_cwt = np.stack(all_cwt, axis=0).astype(np.float32) if all_cwt else np.empty(
        (0, config.NUM_CHANNELS, 0, 0), dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)
    return X_cwt, y, pd.DataFrame(rows)


def _build_td(manifest: pd.DataFrame):
    all_td, all_y, rows = [], [], []
    for w, label_idx, meta in _windows_and_labels(manifest):
        all_td.append(window_to_td_features(w))
        all_y.append(label_idx)
        rows.append(meta)
    X_td = np.stack(all_td, axis=0).astype(np.float32) if all_td else np.empty(
        (0, config.NUM_TD_FEATURES), dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)
    return X_td, y, pd.DataFrame(rows)


def _save_npz(path, X, y, window_manifest):
    np.savez_compressed(
        path, X=X, y=y,
        window_manifest=window_manifest.to_numpy(dtype=object),
        window_manifest_cols=np.array(window_manifest.columns),
    )


def _load_npz(path):
    data = np.load(path, allow_pickle=True)
    window_manifest = pd.DataFrame(data["window_manifest"], columns=data["window_manifest_cols"])
    return data["X"], data["y"], window_manifest


def build_td_split(manifest: pd.DataFrame, split_name: str, use_cache: bool = True
                    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """TD features only -- cheap, used on its own for the classical-model
    search (src/classical.py) so that step doesn't need the much larger CWT
    arrays resident in memory at the same time (holding all five splits'
    CWT tensors -- ~2.7GB -- alongside a parallel, multi-process
    hyperparameter search was observed to crash the process; see README)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    td_path = CACHE_DIR / f"td_{split_name}_{_td_hash()}.npz"
    if use_cache and td_path.exists():
        return _load_npz(td_path)
    X_td, y_td, wm_td = _build_td(manifest)
    if use_cache:
        _save_npz(td_path, X_td, y_td, wm_td)
    return X_td, y_td, wm_td


def build_cwt_split(manifest: pd.DataFrame, split_name: str, use_cache: bool = True
                     ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """CWT scalograms only -- the expensive representation."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cwt_path = CACHE_DIR / f"cwt_{split_name}_{_cwt_hash()}.npz"
    if use_cache and cwt_path.exists():
        return _load_npz(cwt_path)
    X_cwt, y_cwt, wm_cwt = _build_cwt(manifest)
    if use_cache:
        _save_npz(cwt_path, X_cwt, y_cwt, wm_cwt)
    return X_cwt, y_cwt, wm_cwt


def build_split(manifest: pd.DataFrame, split_name: str, use_cache: bool = True
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Returns (X_cwt, X_td, y, window_manifest):
      X_cwt: (N, num_channels, H, W) float32 CWT tensors
      X_td:  (N, num_td_features) float32 classical feature vectors
      y:     (N,) int64 class indices (index into config.GESTURES)
      window_manifest: one row per window, with source file/session/gesture.
    CWT and TD features are cached and invalidated independently (see hash
    functions above) -- changing a TD feature does not force the far more
    expensive CWT scalograms to be recomputed, and vice versa. Loads both at
    once -- use build_td_split/build_cwt_split directly when only one is
    needed (e.g. the classical-model search only needs TD).
    """
    X_cwt, y_cwt, wm_cwt = build_cwt_split(manifest, split_name, use_cache)
    X_td, y_td, wm_td = build_td_split(manifest, split_name, use_cache)

    assert len(X_cwt) == len(X_td), (
        f"CWT ({len(X_cwt)}) and TD ({len(X_td)}) window counts disagree for "
        f"split '{split_name}' -- windowing config must have changed without "
        f"bumping both cache versions."
    )
    assert (y_cwt == y_td).all(), "CWT/TD label mismatch -- windowing order diverged"

    return X_cwt, X_td, y_cwt, wm_cwt


class EMGWindowDataset(Dataset):
    """Standardizes TD features using stats fit on the TRAIN split only
    (passed in as td_mean/td_std), then serves (cwt, td, y) triples."""

    def __init__(self, X_cwt: np.ndarray, X_td: np.ndarray, y: np.ndarray,
                 td_mean: np.ndarray, td_std: np.ndarray):
        self.X_cwt = torch.from_numpy(X_cwt)
        self.X_td = torch.from_numpy((X_td - td_mean) / td_std)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_cwt[idx], self.X_td[idx], self.y[idx]


def fit_td_scaler(X_td_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = X_td_train.mean(axis=0, keepdims=True)
    std = X_td_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)
