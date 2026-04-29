#!/usr/bin/env python
"""Benchmark all feature presets on RF classifier."""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from semg_model import FEATURE_PRESETS, WINDOW_SIZE
from semg_train_rf import create_samples, time_based_split


def eval_preset(
    preset_name,
    df_train,
    df_val,
    unseen_dir,
    n_estimators=400,
    max_depth=20,
    min_samples_split=4,
    min_samples_leaf=2,
):
    """Train RF on preset and return val/unseen accuracy."""
    step_size = WINDOW_SIZE // 4

    X_train, y_train_labels = create_samples(df_train, WINDOW_SIZE, step_size, feature_preset=preset_name)
    X_val, y_val_labels = create_samples(df_val, WINDOW_SIZE, step_size, feature_preset=preset_name)

    le = LabelEncoder()
    le.fit(np.unique(np.concatenate([y_train_labels, y_val_labels])))
    y_train = le.transform(y_train_labels)
    y_val = le.transform(y_val_labels)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    val_acc = float(clf.score(X_val, y_val))

    # Evaluate on unseen
    all_preds, all_labels = [], []
    for csv_path in sorted(glob.glob(os.path.join(unseen_dir, "*.csv"))):
        try:
            d = pd.read_csv(csv_path)
            X_u, y_u = create_samples(d, WINDOW_SIZE, step_size, feature_preset=preset_name)
        except (ValueError, FileNotFoundError):
            continue

        y_u_enc = le.transform(y_u)
        p = clf.predict(X_u)
        all_preds.extend(p)
        all_labels.extend(y_u_enc)

    unseen_acc = float(np.mean(np.array(all_preds) == np.array(all_labels))) if all_preds else np.nan

    return {
        "preset": preset_name,
        "n_features": X_train.shape[1],
        "val_acc": val_acc,
        "unseen_acc": unseen_acc,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark all RF feature presets")
    parser.add_argument("--data", default="../sutd_bmi_safety_data/combined.csv", help="Path to combined CSV")
    parser.add_argument("--unseen", default="../sutd_bmi_safety_data/unseen", help="Folder containing unseen CSV files")
    parser.add_argument("--n_estimators", type=int, default=400, help="Number of RF trees")
    parser.add_argument("--max_depth", type=int, default=20, help="Max tree depth")
    parser.add_argument("--min_samples_split", type=int, default=4, help="Min samples required to split")
    parser.add_argument("--min_samples_leaf", type=int, default=2, help="Min samples per leaf")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--output", default="preset_benchmark_results.csv", help="Output CSV path")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df_train, df_val = time_based_split(df, val_ratio=args.val_ratio)

    presets = sorted(FEATURE_PRESETS.keys())
    results = []

    print(f"Benchmarking {len(presets)} presets...")
    print("=" * 80)

    for i, preset in enumerate(presets, 1):
        print(f"[{i}/{len(presets)}] {preset}...", end=" ", flush=True)
        try:
            result = eval_preset(
                preset,
                df_train,
                df_val,
                args.unseen,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                min_samples_split=args.min_samples_split,
                min_samples_leaf=args.min_samples_leaf,
            )
            results.append(result)
            print(f"✓ {result['n_features']:2d}F | val={result['val_acc']:.4f} | unseen={result['unseen_acc']:.4f}")
        except Exception as e:
            print(f"✗ ERROR: {e}")

    print("=" * 80)
    print("\nResults (sorted by unseen accuracy):")
    print("-" * 80)
    print(f"{'Preset':<35} {'Features':>8} {'Val Acc':>10} {'Unseen Acc':>12}")
    print("-" * 80)

    sorted_results = sorted(results, key=lambda x: x["unseen_acc"], reverse=True)
    for r in sorted_results:
        print(
            f"{r['preset']:<35} {r['n_features']:>8d} "
            f"{r['val_acc']:>10.4f} {r['unseen_acc']:>12.4f}"
        )

    print("\n" + "=" * 80)
    print("Summary:")
    if not results:
        print("No successful preset evaluations. Check input paths and data format.")
        sys.exit(1)

    best_val = max(results, key=lambda x: x["val_acc"])
    best_unseen = max(results, key=lambda x: x["unseen_acc"])
    print(f"Best validation accuracy: {best_val['preset']} ({best_val['val_acc']:.4f})")
    print(f"Best unseen accuracy:     {best_unseen['preset']} ({best_unseen['unseen_acc']:.4f})")

    # Save results to CSV
    df_results = pd.DataFrame(sorted_results)
    df_results.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")
