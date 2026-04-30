"""
Calibration: fine-tune last layer of pretrained model on a small
number of samples from a new session.

Usage:
    # Calibrate using 2 files per gesture, save calibrated model
    python semg_calibrate.py --csv "../data/unseen_session/*.csv" --n_calib 2

    # Then run inference as normal — it will load semg_mlp_calibrated.pth
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import defaultdict

import joblib
from semg_model import SEMGMLP, extract_features, WINDOW_SIZE, SEMGDataset, get_feature_dim

# Load pretrained model and config
le = joblib.load("label_encoder_mlp.pkl")
feature_config = joblib.load("feature_config.pkl")
feature_names = feature_config["feature_names"]

input_dim = get_feature_dim(n_channels=4, feature_names=feature_names)
n_classes = len(le.classes_)


def extract_samples_from_csv(csv_path):
    basename = os.path.basename(csv_path)
    parts = basename.split("_")
    ground_truth = None
    if len(parts) >= 2:
        gesture = parts[1]
        for cls in le.classes_:
            if cls.startswith(gesture):
                ground_truth = cls
                break

    if ground_truth is None:
        return [], []

    df = pd.read_csv(csv_path)
    X, y = [], []
    for i in range(0, len(df), WINDOW_SIZE):
        chunk = df.iloc[i:i + WINDOW_SIZE]
        if len(chunk) < WINDOW_SIZE:
            continue
        sample = chunk[['Ch0 Act', 'Ch1 Act', 'Ch2 Act', 'Ch3 Act']].values.T
        feats = extract_features(sample, feature_names=feature_names)
        X.append(feats)
        y.append(ground_truth)

    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Glob pattern for session CSVs")
    parser.add_argument("--n_calib", type=int, default=2,
                        help="Number of files per gesture for calibration")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--eval", action="store_true",
                        help="Also evaluate on remaining files (for testing)")
    args = parser.parse_args()

    csv_files = sorted(glob.glob(args.csv, recursive=True))
    if not csv_files:
        print("No CSV files found.")
        return

    # Group files by gesture
    gesture_files = defaultdict(list)
    for f in csv_files:
        basename = os.path.basename(f)
        parts = basename.split("_")
        if len(parts) >= 2:
            gesture = parts[1]
            for cls in le.classes_:
                if cls.startswith(gesture):
                    gesture_files[cls].append(f)
                    break

    # Split into calibration and evaluation
    calib_files = []
    eval_files = []
    for cls in sorted(gesture_files.keys()):
        files = gesture_files[cls]
        calib_files.extend(files[:args.n_calib])
        eval_files.extend(files[args.n_calib:])

    print(f"Calibration: {len(calib_files)} files ({args.n_calib} per gesture)")

    # Extract calibration samples
    calib_X, calib_y = [], []
    for f in calib_files:
        X, y = extract_samples_from_csv(f)
        calib_X.extend(X)
        calib_y.extend(y)

    calib_X = np.stack(calib_X)
    calib_y_encoded = le.transform(calib_y)

    # Load pretrained model
    model = SEMGMLP(n_classes, input_dim)
    model.load_state_dict(torch.load("semg_mlp.pth", map_location="cpu"))

    # Freeze all layers except the last two linear layers
    for param in model.parameters():
        param.requires_grad = False
    # # Unfreeze last layer only (layer 3: 64 -> n_classes)
    # last_layer = model.net[-1]
    # for param in last_layer.parameters():
    #     param.requires_grad = True
    # Unfreeze layer 2 (128 -> 64) and layer 3 (64 -> n_classes)
    for layer in model.net[3:]:  # net[3]=Linear(128,64), net[4]=ReLU, net[5]=Dropout, net[6]=Linear(64,n_classes)
        if hasattr(layer, 'parameters'):
            for param in layer.parameters():
                param.requires_grad = True

    # Fine-tune
    calib_ds = SEMGDataset(calib_X, calib_y_encoded)
    calib_loader = DataLoader(calib_ds, batch_size=8, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for Xb, yb in calib_loader:
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * Xb.size(0)
        avg_loss = total_loss / len(calib_ds)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}")

    # Save calibrated model
    torch.save(model.state_dict(), "semg_mlp_calibrated.pth")
    print("\nCalibrated model saved to semg_mlp_calibrated.pth")

    # Evaluate if requested
    if args.eval and eval_files:
        print(f"\nEvaluating on {len(eval_files)} remaining files...")

        eval_X, eval_y = [], []
        for f in eval_files:
            X, y = extract_samples_from_csv(f)
            eval_X.extend(X)
            eval_y.extend(y)
        eval_X = np.stack(eval_X)
        eval_y_encoded = le.transform(eval_y)

        # Before calibration (original model)
        model_pre = SEMGMLP(n_classes, input_dim)
        model_pre.load_state_dict(torch.load("semg_mlp.pth", map_location="cpu"))
        model_pre.eval()
        with torch.no_grad():
            pre_preds = model_pre(torch.tensor(eval_X, dtype=torch.float32)).argmax(dim=1).numpy()

        # After calibration
        model.eval()
        with torch.no_grad():
            post_preds = model(torch.tensor(eval_X, dtype=torch.float32)).argmax(dim=1).numpy()

        pre_acc = (pre_preds == eval_y_encoded).mean()
        post_acc = (post_preds == eval_y_encoded).mean()

        print(f"\n{'Class':<25} {'Before':>8} {'After':>8} {'Change':>8}")
        print("-" * 51)
        for i, cls in enumerate(le.classes_):
            mask = eval_y_encoded == i
            if mask.sum() == 0:
                continue
            pre_cls = (pre_preds[mask] == i).mean()
            post_cls = (post_preds[mask] == i).mean()
            print(f"  {cls:<23} {pre_cls:>8.4f} {post_cls:>8.4f} {post_cls - pre_cls:>+8.4f}")
        print("-" * 51)
        print(f"  {'Overall':<23} {pre_acc:>8.4f} {post_acc:>8.4f} {post_acc - pre_acc:>+8.4f}")


if __name__ == "__main__":
    main()
