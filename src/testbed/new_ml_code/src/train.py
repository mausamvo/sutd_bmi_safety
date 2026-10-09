"""
Trains EMGNetFusion on the baseline train split only, model-selects on the
baseline val split (carved out of the 80% train_pool -- see data.py). The
20% test split and both attack splits stay frozen-model evaluation only.
"""
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config
from .dataset import EMGWindowDataset, fit_td_scaler
from .model import EMGNetFusion


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_emgnet(
    X_cwt_train: np.ndarray, X_td_train: np.ndarray, y_train: np.ndarray,
    X_cwt_val: np.ndarray, X_td_val: np.ndarray, y_val: np.ndarray,
    seed: int = 0,
) -> Tuple[EMGNetFusion, dict, np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    device = get_device()
    model = EMGNetFusion(cnn_dropout=config.CNN_DROPOUT, td_dropout=config.TD_DROPOUT).to(device)

    td_mean, td_std = fit_td_scaler(X_td_train)  # fit on train only, applied everywhere

    train_loader = DataLoader(
        EMGWindowDataset(X_cwt_train, X_td_train, y_train, td_mean, td_std),
        batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(
        EMGWindowDataset(X_cwt_val, X_td_val, y_val, td_mean, td_std),
        batch_size=config.BATCH_SIZE, shuffle=False)

    # The TD branch (over classical hand-crafted features) converges to a much
    # better solution on its own than the CWT-CNN branch does on this small
    # pool (see README) -- a single shared lr=0.01 (tuned for the CNN) was
    # found to destabilize the TD branch's BatchNorm statistics early in
    # training and drag overall fusion accuracy below the TD branch alone.
    # Separate learning rates per branch fixes this: the CNN backbone keeps
    # the paper's lr=0.01, while the TD branch + final classifier train at a
    # gentler rate that lets the TD branch's already-good representation
    # settle rather than get knocked around by the CNN's larger updates.
    optimizer = torch.optim.Adam([
        {"params": model.cnn_backbone.parameters(), "lr": config.INITIAL_LR,
         "weight_decay": config.CNN_WEIGHT_DECAY},
        {"params": model.td_branch.parameters(), "lr": config.TD_BRANCH_LR,
         "weight_decay": config.TD_WEIGHT_DECAY},
        {"params": model.classifier.parameters(), "lr": config.TD_BRANCH_LR,
         "weight_decay": config.TD_WEIGHT_DECAY},
    ])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(config.LR_DECAY_EPOCHS), gamma=1.0 / config.LR_DECAY_FACTOR)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for cwt_b, td_b, yb in train_loader:
            cwt_b, td_b, yb = cwt_b.to(device), td_b.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(cwt_b, td_b)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * yb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
        train_loss, train_acc = running_loss / total, correct / total

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for cwt_b, td_b, yb in val_loader:
                cwt_b, td_b, yb = cwt_b.to(device), td_b.to(device), yb.to(device)
                logits = model(cwt_b, td_b)
                loss = criterion(logits, yb)
                v_loss += loss.item() * yb.size(0)
                v_correct += (logits.argmax(1) == yb).sum().item()
                v_total += yb.size(0)
        val_loss, val_acc = v_loss / v_total, v_correct / v_total
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0 or epoch == config.NUM_EPOCHS:
            print(f"epoch {epoch:3d}/{config.NUM_EPOCHS}  "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    model.load_state_dict(best_state)
    print(f"Loaded best-val-accuracy checkpoint: val_acc={best_val_acc:.4f}")
    return model, history, td_mean, td_std
