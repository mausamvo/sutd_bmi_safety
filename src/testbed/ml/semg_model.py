try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset
except ImportError:
    torch = None

    class _MissingLayer:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for neural network model classes in semg_model.py")

    class _NNPlaceholder:
        Module = object
        Conv1d = _MissingLayer
        BatchNorm1d = _MissingLayer
        ReLU = _MissingLayer
        MaxPool1d = _MissingLayer
        Linear = _MissingLayer

    nn = _NNPlaceholder()

    class Dataset:
        pass

import numpy as np

# Parameters
WINDOW_SIZE = 100  # this is roughly 1 second

# ---------------------------
# Models
# ---------------------------
class SEMGCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)

        self.fc1 = nn.Linear(32 * (WINDOW_SIZE // 2), 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

class SEMGDataset(Dataset):
    def __init__(self, X, y):
        if torch is None:
            raise ImportError("PyTorch is required for SEMGDataset")
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SEMGMLP(nn.Module):
    def __init__(self, n_classes, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, n_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ---------------------------
# Feature presets
# ---------------------------
# BASELINE FEATURE GROUPS (always computed for 4 channels):
#   MAV (Mean Absolute Value):     Measures average signal amplitude → reflects muscle activation intensity
#   RMS (Root Mean Square):        Represents signal power → correlated with muscle contraction strength
#   WL (Waveform Length):          Cumulative signal changes → indicates muscle fatigue & activity complexity
#   VAR (Variance):                Spread of signal energy → distinguishes static vs dynamic muscle states
#   ZC (Zero Crossing):            Counts zero-crossings → related to muscle fibre firing frequency
#   SSC (Slope Sign Changes):      Counts sign changes in signal slope → detects rapid muscle activation changes
#
# ADDITIONAL FEATURE GROUPS (controlled by config):
#   WAMP (Willison Amplitude):     Counts samples exceeding threshold → reflects muscle recruitment
#   IEMG (Integrated EMG):         Sum of absolute signal values → cumulative muscle activity
#   FREQ (Frequency domain):       mean_freq, median_freq, spectral_entropy → captures spectral characteristics
#   DISTRIBUTION:                  std, peak-to-peak → signal variability metrics
#
# Config keys:
#   add_wamp, add_iemg, add_freq, add_distribution: boolean feature groups
#   zc_threshold, ssc_threshold, wamp_threshold: feature-specific thresholds
#   add_zc, add_ssc: disable baseline zero-crossing / slope-sign-change (default True)
FEATURE_PRESETS = {
    "baseline": {
        "add_wamp": False,
        "add_iemg": False,
        "add_freq": False,
        "add_distribution": False,
        "add_zc": True,
        "add_ssc": True,
        "zc_threshold": 0.01,
        "ssc_threshold": 0.01,
        "wamp_threshold": 0.02,
    },
    "baseline_plus_wamp": {
        "add_wamp": True,
        "add_iemg": False,
        "add_freq": False,
        "add_distribution": False,
        "add_zc": True,
        "add_ssc": True,
        "zc_threshold": 0.01,
        "ssc_threshold": 0.01,
        "wamp_threshold": 0.02,
    },
    "rf_enhanced": {
        "add_wamp": True,
        "add_iemg": True,
        "add_freq": True,
        "add_distribution": True,
        "add_zc": True,
        "add_ssc": True,
        "zc_threshold": 0.008,
        "ssc_threshold": 0.008,
        "wamp_threshold": 0.015,
    },
    # rf_enhanced single-group removals
    "rf_enhanced_no_wamp": {
        "add_wamp": False,
        "add_iemg": True,
        "add_freq": True,
        "add_distribution": True,
        "add_zc": True,
        "add_ssc": True,
        "zc_threshold": 0.008,
        "ssc_threshold": 0.008,
        "wamp_threshold": 0.015,
    },
    "rf_enhanced_no_iemg": {
        "add_wamp": True,
        "add_iemg": False,
        "add_freq": True,
        "add_distribution": True,
        "add_zc": True,
        "add_ssc": True,
        "zc_threshold": 0.008,
        "ssc_threshold": 0.008,
        "wamp_threshold": 0.015,
    },
    "rf_enhanced_no_freq": {
        "add_wamp": True,
        "add_iemg": True,
        "add_freq": False,
        "add_distribution": True,
        "add_zc": True,
        "add_ssc": True,
        "zc_threshold": 0.008,
        "ssc_threshold": 0.008,
        "wamp_threshold": 0.015,
    },
    "rf_enhanced_no_distribution": {
        "add_wamp": True,
        "add_iemg": True,
        "add_freq": True,
        "add_distribution": False,
        "add_zc": True,
        "add_ssc": True,
        "zc_threshold": 0.008,
        "ssc_threshold": 0.008,
        "wamp_threshold": 0.015,
    },
    "rf_enhanced_no_zc": {
        "add_wamp": True,
        "add_iemg": True,
        "add_freq": True,
        "add_distribution": True,
        "add_zc": False,
        "add_ssc": True,
        "zc_threshold": 0.008,
        "ssc_threshold": 0.008,
        "wamp_threshold": 0.015,
    },
    "rf_enhanced_no_ssc": {
        "add_wamp": True,
        "add_iemg": True,
        "add_freq": True,
        "add_distribution": True,
        "add_zc": True,
        "add_ssc": False,
        "zc_threshold": 0.008,
        "ssc_threshold": 0.008,
        "wamp_threshold": 0.015,
    },
    # Multi-group removals (strategic combinations)
    "rf_enhanced_no_zc_median": {
        "add_wamp": True,
        "add_iemg": True,
        "add_freq": True,
        "add_distribution": True,
        "add_zc": False,
        "add_ssc": True,
        "zc_threshold": 0.008,
        "ssc_threshold": 0.008,
        "wamp_threshold": 0.015,
        "drop_median_freq": True,
    },
    "rf_enhanced_no_freq_no_distribution": {
        "add_wamp": True,
        "add_iemg": True,
        "add_freq": False,
        "add_distribution": False,
        "add_zc": True,
        "add_ssc": True,
        "zc_threshold": 0.008,
        "ssc_threshold": 0.008,
        "wamp_threshold": 0.015,
    },
    "rf_enhanced_core": {
        "add_wamp": True,
        "add_iemg": True,
        "add_freq": False,
        "add_distribution": False,
        "add_zc": True,
        "add_ssc": True,
        "zc_threshold": 0.008,
        "ssc_threshold": 0.008,
        "wamp_threshold": 0.015,
    },
    "rf_enhanced_light": {
        "add_wamp": True,
        "add_iemg": False,
        "add_freq": False,
        "add_distribution": False,
        "add_zc": True,
        "add_ssc": True,
        "zc_threshold": 0.008,
        "ssc_threshold": 0.008,
        "wamp_threshold": 0.015,
    },
    # Single baseline feature removal tests
    "baseline_no_mav": {
        "add_wamp": False,
        "add_iemg": False,
        "add_freq": False,
        "add_distribution": False,
        "add_mav": False,
        "add_rms": True,
        "add_wl": True,
        "add_var": True,
        "add_zc": True,
        "add_ssc": True,
        "zc_threshold": 0.01,
        "ssc_threshold": 0.01,
        "wamp_threshold": 0.02,
    },
    "baseline_no_rms": {
        "add_wamp": False,
        "add_iemg": False,
        "add_freq": False,
        "add_distribution": False,
        "add_mav": True,
        "add_rms": False,
        "add_wl": True,
        "add_var": True,
        "add_zc": True,
        "add_ssc": True,
        "zc_threshold": 0.01,
        "ssc_threshold": 0.01,
        "wamp_threshold": 0.02,
    },
    "baseline_no_wl": {
        "add_wamp": False,
        "add_iemg": False,
        "add_freq": False,
        "add_distribution": False,
        "add_mav": True,
        "add_rms": True,
        "add_wl": False,
        "add_var": True,
        "add_zc": True,
        "add_ssc": True,
        "zc_threshold": 0.01,
        "ssc_threshold": 0.01,
        "wamp_threshold": 0.02,
    },
    "baseline_no_var": {
        "add_wamp": False,
        "add_iemg": False,
        "add_freq": False,
        "add_distribution": False,
        "add_mav": True,
        "add_rms": True,
        "add_wl": True,
        "add_var": False,
        "add_zc": True,
        "add_ssc": True,
        "zc_threshold": 0.01,
        "ssc_threshold": 0.01,
        "wamp_threshold": 0.02,
    },
    "baseline_no_zc_only": {
        "add_wamp": False,
        "add_iemg": False,
        "add_freq": False,
        "add_distribution": False,
        "add_mav": True,
        "add_rms": True,
        "add_wl": True,
        "add_var": True,
        "add_zc": False,
        "add_ssc": True,
        "zc_threshold": 0.01,
        "ssc_threshold": 0.01,
        "wamp_threshold": 0.02,
    },
    "baseline_no_ssc_only": {
        "add_wamp": False,
        "add_iemg": False,
        "add_freq": False,
        "add_distribution": False,
        "add_mav": True,
        "add_rms": True,
        "add_wl": True,
        "add_var": True,
        "add_zc": True,
        "add_ssc": False,
        "zc_threshold": 0.01,
        "ssc_threshold": 0.01,
        "wamp_threshold": 0.02,
    },
    # Multi-baseline feature removal tests
    "baseline_core4": {
        "add_wamp": False,
        "add_iemg": False,
        "add_freq": False,
        "add_distribution": False,
        "add_mav": True,
        "add_rms": True,
        "add_wl": True,
        "add_var": True,
        "add_zc": False,
        "add_ssc": False,
        "zc_threshold": 0.01,
        "ssc_threshold": 0.01,
        "wamp_threshold": 0.02,
    },
    "baseline_amp_only": {
        "add_wamp": False,
        "add_iemg": False,
        "add_freq": False,
        "add_distribution": False,
        "add_mav": True,
        "add_rms": True,
        "add_wl": False,
        "add_var": False,
        "add_zc": False,
        "add_ssc": False,
        "zc_threshold": 0.01,
        "ssc_threshold": 0.01,
        "wamp_threshold": 0.02,
    },
    "baseline_mav_var_only": {
        "add_wamp": False,
        "add_iemg": False,
        "add_freq": False,
        "add_distribution": False,
        "add_mav": True,
        "add_rms": False,
        "add_wl": False,
        "add_var": True,
        "add_zc": False,
        "add_ssc": False,
        "zc_threshold": 0.01,
        "ssc_threshold": 0.01,
        "wamp_threshold": 0.02,
    },
    "baseline_no_wl_no_zc": {
        "add_wamp": False,
        "add_iemg": False,
        "add_freq": False,
        "add_distribution": False,
        "add_mav": True,
        "add_rms": True,
        "add_wl": False,
        "add_var": True,
        "add_zc": False,
        "add_ssc": True,
        "zc_threshold": 0.01,
        "ssc_threshold": 0.01,
        "wamp_threshold": 0.02,
    },
    "baseline_no_wl_no_ssc": {
        "add_wamp": False,
        "add_iemg": False,
        "add_freq": False,
        "add_distribution": False,
        "add_mav": True,
        "add_rms": True,
        "add_wl": False,
        "add_var": True,
        "add_zc": True,
        "add_ssc": False,
        "zc_threshold": 0.01,
        "ssc_threshold": 0.01,
        "wamp_threshold": 0.02,
    },
    "baseline_no_wl_no_rms": {
        "add_wamp": False,
        "add_iemg": False,
        "add_freq": False,
        "add_distribution": False,
        "add_mav": True,
        "add_rms": False,
        "add_wl": False,
        "add_var": True,
        "add_zc": True,
        "add_ssc": True,
        "zc_threshold": 0.01,
        "ssc_threshold": 0.01,
        "wamp_threshold": 0.02,
    },
    "baseline_no_wl_no_mav": {
        "add_wamp": False,
        "add_iemg": False,
        "add_freq": False,
        "add_distribution": False,
        "add_mav": False,
        "add_rms": True,
        "add_wl": False,
        "add_var": True,
        "add_zc": True,
        "add_ssc": True,
        "zc_threshold": 0.01,
        "ssc_threshold": 0.01,
        "wamp_threshold": 0.02,
    },
    "baseline_no_wl_no_var": {
        "add_wamp": False,
        "add_iemg": False,
        "add_freq": False,
        "add_distribution": False,
        "add_mav": True,
        "add_rms": True,
        "add_wl": False,
        "add_var": False,
        "add_zc": True,
        "add_ssc": True,
        "zc_threshold": 0.01,
        "ssc_threshold": 0.01,
        "wamp_threshold": 0.02,
    },
}

# ---------------------------
# Feature functions
# ---------------------------
def _mav(x):
    return np.mean(np.abs(x))

def _rms(x):
    return np.sqrt(np.mean(x ** 2))

def _wl(x):
    return np.sum(np.abs(np.diff(x)))

def _var(x):
    return np.var(x)

def _zc(x, threshold=0.01):
    return np.sum(
        ((x[:-1] * x[1:]) < 0) &
        (np.abs(x[:-1] - x[1:]) > threshold)
    )

def _ssc(x, threshold=0.01):
    return np.sum(
        ((x[1:-1] - x[:-2]) * (x[1:-1] - x[2:]) > 0) &
        (np.abs(x[1:-1] - x[:-2]) > threshold) &
        (np.abs(x[1:-1] - x[2:]) > threshold)
    )

def _wamp(x, threshold=0.02):
    return np.sum(np.abs(np.diff(x)) > threshold)

def _iemg(x):
    return np.sum(np.abs(x))

def _std(x):
    return np.std(x)

def _peak_to_peak(x):
    return np.ptp(x)

def _mean_freq(x):
    fft_vals = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x))
    power = fft_vals ** 2
    total_power = np.sum(power) + 1e-10
    return np.sum(freqs * power) / total_power

def _median_freq(x):
    fft_vals = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x))
    power = fft_vals ** 2
    cumulative = np.cumsum(power)
    half_power = cumulative[-1] / 2.0 if len(cumulative) > 0 else 0.0
    idx = int(np.searchsorted(cumulative, half_power)) if len(cumulative) > 0 else 0
    if len(freqs) == 0:
        return 0.0
    idx = min(idx, len(freqs) - 1)
    return freqs[idx]

def _spectral_entropy(x):
    fft_vals = np.abs(np.fft.rfft(x))
    power = fft_vals ** 2
    psd = power / (np.sum(power) + 1e-10)
    return -np.sum(psd * np.log2(psd + 1e-12))


def get_feature_names(preset="baseline", n_channels=4, config=None):
    if config is None:
        if preset not in FEATURE_PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Available: {list(FEATURE_PRESETS.keys())}")
        config = FEATURE_PRESETS[preset]

    names = []
    baseline_names = []
    if config.get("add_mav", True):
        baseline_names.append("mav")
    if config.get("add_rms", True):
        baseline_names.append("rms")
    if config.get("add_wl", True):
        baseline_names.append("wl")
    if config.get("add_var", True):
        baseline_names.append("var")
    if config.get("add_zc", True):
        baseline_names.append("zc")
    if config.get("add_ssc", True):
        baseline_names.append("ssc")

    for ch_idx in range(n_channels):
        names.extend([f"ch{ch_idx}_{name}" for name in baseline_names])

    if config.get("add_wamp", False):
        names.extend([f"ch{ch_idx}_wamp" for ch_idx in range(n_channels)])
    if config.get("add_iemg", False):
        names.extend([f"ch{ch_idx}_iemg" for ch_idx in range(n_channels)])
    if config.get("add_freq", False):
        names.extend([f"ch{ch_idx}_mean_freq" for ch_idx in range(n_channels)])
        if not config.get("drop_median_freq", False):
            names.extend([f"ch{ch_idx}_median_freq" for ch_idx in range(n_channels)])
        names.extend([f"ch{ch_idx}_spectral_entropy" for ch_idx in range(n_channels)])
    if config.get("add_distribution", False):
        names.extend([f"ch{ch_idx}_std" for ch_idx in range(n_channels)])
        names.extend([f"ch{ch_idx}_ptp" for ch_idx in range(n_channels)])

    return names

# ---------------------------
# Baseline features (same as your current)
# ---------------------------
def extract_features_baseline(window, zc_threshold=0.01, ssc_threshold=0.01, add_zc=True, add_ssc=True,
                               add_mav=True, add_rms=True, add_wl=True, add_var=True):
    feats = []
    for ch in window:
        baseline_feats = []
        if add_mav:
            baseline_feats.append(_mav(ch))
        if add_rms:
            baseline_feats.append(_rms(ch))
        if add_wl:
            baseline_feats.append(_wl(ch))
        if add_var:
            baseline_feats.append(_var(ch))
        feats.extend(baseline_feats)
        if add_zc:
            feats.append(_zc(ch, threshold=zc_threshold))
        if add_ssc:
            feats.append(_ssc(ch, threshold=ssc_threshold))
    return np.array(feats, dtype=np.float32)

# ---------------------------
# Main feature extractor
# ---------------------------
def extract_features(window, preset="baseline", config=None):
    if config is None:
        if preset not in FEATURE_PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Available: {list(FEATURE_PRESETS.keys())}")
        config = FEATURE_PRESETS[preset]

    feats = extract_features_baseline(
        window,
        zc_threshold=config.get("zc_threshold", 0.01),
        ssc_threshold=config.get("ssc_threshold", 0.01),
        add_zc=config.get("add_zc", True),
        add_ssc=config.get("add_ssc", True),
        add_mav=config.get("add_mav", True),
        add_rms=config.get("add_rms", True),
        add_wl=config.get("add_wl", True),
        add_var=config.get("add_var", True),
    ).tolist()

    if config.get("add_wamp", False):
        wamp_threshold = config.get("wamp_threshold", 0.02)
        for ch in window:
            feats.append(_wamp(ch, threshold=wamp_threshold))

    if config.get("add_iemg", False):
        for ch in window:
            feats.append(_iemg(ch))

    if config.get("add_freq", False):
        for ch in window:
            feats.append(_mean_freq(ch))
        if not config.get("drop_median_freq", False):
            for ch in window:
                feats.append(_median_freq(ch))
        for ch in window:
            feats.append(_spectral_entropy(ch))

    if config.get("add_distribution", False):
        for ch in window:
            feats.append(_std(ch))
        for ch in window:
            feats.append(_peak_to_peak(ch))

    return np.array(feats, dtype=np.float32)