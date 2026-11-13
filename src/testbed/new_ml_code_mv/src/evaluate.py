"""
Frozen-model evaluation on a given split: baseline val (during training,
for checkpoint selection), and -- after training is complete -- baseline
test (the headline 20% split) plus both attack conditions.
"""
from typing import Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from . import config
from .dataset import EMGWindowDataset
from .model import EMGNetFusion
from .train import get_device


def _per_file_majority_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                                 window_manifest: pd.DataFrame) -> Dict:
    """
    Every window in a recording file shares the same ground-truth gesture
    (a file IS one gesture repetition). The paper's own Appendix A describes
    exactly this: smoothing predictions over consecutive windows within one
    gesture hold, since single-window misclassifications are noise that a
    real-time deployment would filter out. This computes that per-file
    majority-vote accuracy as a second, complementary metric alongside raw
    per-window accuracy -- not a replacement for it.
    """
    if len(window_manifest) == 0:
        return {"file_majority_accuracy": float("nan"), "n_files": 0}
    df = window_manifest.copy()
    df["y_true"] = y_true
    df["y_pred"] = y_pred
    correct = 0
    for _, g in df.groupby("path"):
        majority_pred = g["y_pred"].mode().iloc[0]
        correct += int(majority_pred == g["y_true"].iloc[0])
    n_files = df["path"].nunique()
    return {"file_majority_accuracy": correct / n_files, "n_files": int(n_files)}


def predict_proba_nn(model: EMGNetFusion, X_cwt: np.ndarray, X_td: np.ndarray,
                      td_mean: np.ndarray, td_std: np.ndarray) -> np.ndarray:
    """Per-window class probabilities from EMGNetFusion (softmax over logits),
    used to build the NN+RandomForest ensemble in run_pipeline.py."""
    device = get_device()
    model = model.to(device)
    model.eval()
    loader = DataLoader(EMGWindowDataset(X_cwt, X_td, np.zeros(len(X_cwt), dtype=np.int64),
                                          td_mean, td_std),
                         batch_size=config.BATCH_SIZE, shuffle=False)
    all_proba = []
    with torch.no_grad():
        for cwt_b, td_b, _ in loader:
            cwt_b, td_b = cwt_b.to(device), td_b.to(device)
            probs = torch.softmax(model(cwt_b, td_b), dim=1)
            all_proba.append(probs.cpu().numpy())
    return np.concatenate(all_proba, axis=0) if all_proba else np.empty((0, config.NUM_CLASSES))


def accuracy_from_proba(proba: np.ndarray, y_true: np.ndarray,
                         window_manifest: pd.DataFrame) -> Dict:
    """Same metrics as evaluate_split (window + file-majority accuracy,
    per-class accuracy, confusion matrix), but starting from a probability
    array instead of running a model -- used for the RandomForest-only and
    NN+RandomForest ensemble evaluations in run_pipeline.py."""
    y_pred = proba.argmax(axis=1)
    overall_acc = accuracy_score(y_true, y_pred) if len(y_true) else float("nan")
    report = classification_report(
        y_true, y_pred, labels=list(range(config.NUM_CLASSES)),
        target_names=config.GESTURES, output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLASSES)))
    per_class_acc = {}
    for i, gesture in enumerate(config.GESTURES):
        mask = y_true == i
        per_class_acc[gesture] = float((y_pred[mask] == i).mean()) if mask.sum() else float("nan")
    result = {
        "n_windows": int(len(y_true)),
        "overall_accuracy": float(overall_acc),
        "per_class_accuracy": per_class_acc,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }
    result.update(_per_file_majority_accuracy(y_true, y_pred, window_manifest))
    return result


def evaluate_split(model: EMGNetFusion, X_cwt: np.ndarray, X_td: np.ndarray, y: np.ndarray,
                    td_mean: np.ndarray, td_std: np.ndarray,
                    window_manifest: pd.DataFrame = None) -> Dict:
    device = get_device()
    model = model.to(device)
    model.eval()

    loader = DataLoader(EMGWindowDataset(X_cwt, X_td, y, td_mean, td_std),
                         batch_size=config.BATCH_SIZE, shuffle=False)
    all_preds, all_true = [], []
    with torch.no_grad():
        for cwt_b, td_b, yb in loader:
            cwt_b, td_b = cwt_b.to(device), td_b.to(device)
            logits = model(cwt_b, td_b)
            preds = logits.argmax(1).cpu().numpy()
            all_preds.append(preds)
            all_true.append(yb.numpy())

    y_pred = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.int64)
    y_true = np.concatenate(all_true) if all_true else np.array([], dtype=np.int64)

    overall_acc = accuracy_score(y_true, y_pred) if len(y_true) else float("nan")
    report = classification_report(
        y_true, y_pred, labels=list(range(config.NUM_CLASSES)),
        target_names=config.GESTURES, output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLASSES)))

    per_class_acc = {}
    for i, gesture in enumerate(config.GESTURES):
        mask = y_true == i
        per_class_acc[gesture] = float((y_pred[mask] == i).mean()) if mask.sum() else float("nan")

    result = {
        "n_windows": int(len(y_true)),
        "overall_accuracy": float(overall_acc),
        "per_class_accuracy": per_class_acc,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }
    if window_manifest is not None:
        result.update(_per_file_majority_accuracy(y_true, y_pred, window_manifest))
    return result
