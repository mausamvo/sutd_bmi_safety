import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset 

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
FEATURE_PRESETS = {
    "baseline": {
        "add_wamp": False,
        "zc_threshold": 0.01,
        "ssc_threshold": 0.01,
        "wamp_threshold": 0.02,
    },
    "baseline_plus_wamp": {
        "add_wamp": True,
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

# ---------------------------
# Baseline features (same as your current)
# ---------------------------
def extract_features_baseline(window, zc_threshold=0.01, ssc_threshold=0.01):
    feats = []
    for ch in window:
        feats.extend([
            _mav(ch),
            _rms(ch),
            _wl(ch),
            _var(ch),
            _zc(ch, threshold=zc_threshold),
            _ssc(ch, threshold=ssc_threshold),
        ])
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
    ).tolist()

    if config.get("add_wamp", False):
        wamp_threshold = config.get("wamp_threshold", 0.02)
        for ch in window:
            feats.append(_wamp(ch, threshold=wamp_threshold))

    return np.array(feats, dtype=np.float32)