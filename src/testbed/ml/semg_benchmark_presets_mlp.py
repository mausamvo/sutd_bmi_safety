#!/usr/bin/env python
"""Benchmark all feature presets on MLP classifier."""

import argparse
import glob
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from semg_model import FEATURE_PRESETS, SEMGMLP, WINDOW_SIZE
from semg_train_rf import create_samples, time_based_split


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def eval_preset(
    preset_name,
    df_train,
    df_val,
    unseen_dir,
    epochs=100,
    batch_size=16,
    lr=1e-3,
    patience=15,
    seed=42,
):
    """Train MLP on a preset and return validation/unseen accuracy."""
    set_seed(seed)

    step_size = WINDOW_SIZE // 4

    X_train, y_train_labels = create_samples(df_train, WINDOW_SIZE, step_size, feature_preset=preset_name)
    X_val, y_val_labels = create_samples(df_val, WINDOW_SIZE, step_size, feature_preset=preset_name)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    le = LabelEncoder()
    le.fit(np.unique(np.concatenate([y_train_labels, y_val_labels])))
    y_train = le.transform(y_train_labels)
    y_val = le.transform(y_val_labels)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SEMGMLP(n_classes=len(le.classes_), input_dim=X_train.shape[1]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_state = None
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)

        val_acc = correct / total if total else 0.0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            break

    if best_state is None:
        raise RuntimeError("No valid MLP state captured during training.")

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    # Unseen evaluation
    all_preds, all_labels = [], []
    for csv_path in sorted(glob.glob(os.path.join(unseen_dir, "*.csv"))):
        try:
            d = pd.read_csv(csv_path)
            X_u, y_u = create_samples(d, WINDOW_SIZE, step_size, feature_preset=preset_name)
        except (ValueError, FileNotFoundError):
            continue

        y_u_enc = le.transform(y_u)
        X_u = scaler.transform(X_u)

        unseen_ds = TensorDataset(
            torch.tensor(X_u, dtype=torch.float32),
            torch.tensor(y_u_enc, dtype=torch.long),
        )
        unseen_loader = DataLoader(unseen_ds, batch_size=batch_size)

        with torch.no_grad():
            for xb, yb in unseen_loader:
                xb = xb.to(device)
                pred = model(xb).argmax(dim=1).cpu().numpy()
                all_preds.extend(pred.tolist())
                all_labels.extend(yb.numpy().tolist())

    unseen_acc = float(np.mean(np.array(all_preds) == np.array(all_labels))) if all_preds else np.nan

    return {
        "preset": preset_name,
        "n_features": int(X_train.shape[1]),
        "val_acc": float(best_val_acc),
        "unseen_acc": float(unseen_acc),
        "best_epoch": int(best_epoch),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark all MLP feature presets")
    parser.add_argument("--data", default="../sutd_bmi_safety_data/combined.csv", help="Path to combined CSV")
    parser.add_argument("--unseen", default="../sutd_bmi_safety_data/unseen", help="Folder containing unseen CSV files")
    parser.add_argument("--epochs", type=int, default=100, help="Max MLP epochs per preset")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default="preset_benchmark_results_mlp.csv", help="Output CSV path")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df_train, df_val = time_based_split(df, val_ratio=args.val_ratio)

    presets = sorted(FEATURE_PRESETS.keys())
    results = []

    print(f"Benchmarking {len(presets)} presets (MLP)...")
    print("=" * 90)

    for i, preset in enumerate(presets, 1):
        print(f"[{i}/{len(presets)}] {preset}...", end=" ", flush=True)
        try:
            result = eval_preset(
                preset,
                df_train,
                df_val,
                args.unseen,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                patience=args.patience,
                seed=args.seed,
            )
            results.append(result)
            print(
                f"✓ {result['n_features']:2d}F | val={result['val_acc']:.4f} "
                f"| unseen={result['unseen_acc']:.4f} | epoch={result['best_epoch']}"
            )
        except Exception as e:
            print(f"✗ ERROR: {e}")

    print("=" * 90)
    print("\nResults (sorted by unseen accuracy):")
    print("-" * 90)
    print(f"{'Preset':<35} {'Features':>8} {'Val Acc':>10} {'Unseen Acc':>12} {'Best Ep':>8}")
    print("-" * 90)

    if not results:
        print("No successful preset evaluations. Check input paths and data format.")
        sys.exit(1)

    sorted_results = sorted(results, key=lambda x: x["unseen_acc"], reverse=True)
    for r in sorted_results:
        print(
            f"{r['preset']:<35} {r['n_features']:>8d} {r['val_acc']:>10.4f} "
            f"{r['unseen_acc']:>12.4f} {r['best_epoch']:>8d}"
        )

    print("\n" + "=" * 90)
    print("Summary:")
    best_val = max(results, key=lambda x: x["val_acc"])
    best_unseen = max(results, key=lambda x: x["unseen_acc"])
    print(f"Best validation accuracy: {best_val['preset']} ({best_val['val_acc']:.4f})")
    print(f"Best unseen accuracy:     {best_unseen['preset']} ({best_unseen['unseen_acc']:.4f})")

    df_results = pd.DataFrame(sorted_results)
    df_results.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")
