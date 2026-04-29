"""Train a Random Forest classifier on sEMG feature data."""

import argparse
import glob
import os
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from semg_model import FEATURE_PRESETS, WINDOW_SIZE, extract_features, get_feature_names


def create_samples(df, window_size, step, feature_preset="baseline"):
    window_size = int(window_size)
    step = int(step)
    X, y = [], []
    for i in range(0, len(df) - window_size + 1, step):
        chunk = df.iloc[i : i + window_size]
        label = chunk["Action"].iloc[0]
        if (chunk["Action"] != label).any():
            continue
        sample = chunk[["Ch0 Act", "Ch1 Act", "Ch2 Act", "Ch3 Act"]].to_numpy().T
        if sample.shape != (4, window_size):
            continue
        feats = extract_features(sample, preset=feature_preset)
        X.append(feats)
        y.append(label)
    if len(X) == 0:
        raise ValueError("No valid windows created.")
    return np.stack(X), np.array(y)


def time_based_split(df, val_ratio=0.2):
    """Split each contiguous action segment chronologically (no leakage)."""
    train_parts, val_parts = [], []
    segment_id = (df["Action"] != df["Action"].shift()).cumsum()
    for _, segment in df.groupby(segment_id):
        n = len(segment)
        split_idx = int(n * (1 - val_ratio))
        if split_idx > 0:
            train_parts.append(segment.iloc[:split_idx])
        if split_idx < n:
            val_parts.append(segment.iloc[split_idx:])
    return pd.concat(train_parts, ignore_index=True), pd.concat(val_parts, ignore_index=True)


def top_feature_importance(clf, feature_names, top_k=12):
    importances = clf.feature_importances_
    order = np.argsort(importances)[::-1]
    rows = []
    for idx in order[: min(top_k, len(order))]:
        rows.append(
            {
                "feature": feature_names[idx] if idx < len(feature_names) else f"f{idx}",
                "importance": float(importances[idx]),
            }
        )
    return pd.DataFrame(rows)


def top_confusions(cm_df, top_k=12):
    rows = []
    labels = list(cm_df.index)
    for actual in labels:
        for predicted in labels:
            if actual == predicted:
                continue
            count = int(cm_df.loc[actual, predicted])
            if count > 0:
                rows.append({"actual": actual, "predicted": predicted, "count": count})
    if not rows:
        return pd.DataFrame(columns=["actual", "predicted", "count"])
    return pd.DataFrame(rows).sort_values("count", ascending=False).head(top_k)


def analyze_bad_leaves(clf, X, y_true, y_pred, feature_names, top_k=12, min_support=6):
    """Find high-error leaves (branches) and split features used on misclassified paths."""
    mis_mask = y_true != y_pred
    leaf_rows = []
    branch_feature_counter = Counter()

    for tree_idx, estimator in enumerate(clf.estimators_):
        tree = estimator.tree_
        leaf_ids = estimator.apply(X)
        decision_paths = estimator.decision_path(X)

        leaves = np.unique(leaf_ids)
        for leaf_id in leaves:
            sample_idx = np.where(leaf_ids == leaf_id)[0]
            support = len(sample_idx)
            if support < min_support:
                continue

            mis_count = int(np.sum(mis_mask[sample_idx]))
            if mis_count == 0:
                continue

            leaf_rows.append(
                {
                    "tree": tree_idx,
                    "leaf": int(leaf_id),
                    "support": int(support),
                    "misclassified": mis_count,
                    "error_rate": float(mis_count / support),
                }
            )

            for i in sample_idx:
                if not mis_mask[i]:
                    continue
                node_start = decision_paths.indptr[i]
                node_end = decision_paths.indptr[i + 1]
                node_ids = decision_paths.indices[node_start:node_end]
                for node_id in node_ids:
                    feat_idx = tree.feature[node_id]
                    if feat_idx >= 0:
                        feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"f{feat_idx}"
                        branch_feature_counter[feat_name] += 1

    if leaf_rows:
        leaf_df = (
            pd.DataFrame(leaf_rows)
            .sort_values(["error_rate", "misclassified", "support"], ascending=[False, False, False])
            .head(top_k)
        )
    else:
        leaf_df = pd.DataFrame(columns=["tree", "leaf", "support", "misclassified", "error_rate"])

    branch_feature_df = pd.DataFrame(
        [
            {"feature": feat, "path_hits_on_misclassified": count}
            for feat, count in branch_feature_counter.most_common(top_k)
        ]
    )
    if branch_feature_df.empty:
        branch_feature_df = pd.DataFrame(columns=["feature", "path_hits_on_misclassified"])

    return leaf_df, branch_feature_df


if __name__ == "__main__":
    if os.path.exists("semg_rf.pkl"):
        os.remove("semg_rf.pkl")
        print("Deleted existing semg_rf.pkl")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="combined.csv")
    parser.add_argument(
        "--feature_preset",
        default="rf_enhanced",
        choices=list(FEATURE_PRESETS.keys()),
    )
    parser.add_argument("--unseen", default=None, help="Path to unseen CSV data for evaluation after training")
    parser.add_argument("--n_estimators", type=int, default=400)
    parser.add_argument("--max_depth", type=int, default=20)
    parser.add_argument("--min_samples_split", type=int, default=4)
    parser.add_argument("--min_samples_leaf", type=int, default=2)
    parser.add_argument("--max_features", default="sqrt", choices=["sqrt", "log2", "none"])
    parser.add_argument("--class_weight", default="balanced_subsample", choices=["balanced", "balanced_subsample", "none"])
    parser.add_argument(
        "--drop_features",
        default="",
        help="Comma-separated feature name substrings to drop (e.g. 'zc,median_freq')",
    )
    parser.add_argument("--top_k_diagnostics", type=int, default=12)
    parser.add_argument("--min_leaf_support", type=int, default=6)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    step_size = int(WINDOW_SIZE // 4)

    df_train, df_val = time_based_split(df, val_ratio=0.2)
    print(f"Time-based split: {len(df_train)} train rows, {len(df_val)} val rows (no leakage)")

    X_train, y_train_labels = create_samples(df_train, WINDOW_SIZE, step_size, feature_preset=args.feature_preset)
    X_test, y_test_labels = create_samples(df_val, WINDOW_SIZE, step_size, feature_preset=args.feature_preset)

    print("Feature preset:", args.feature_preset)
    print("Feature dimension:", X_train.shape[1])

    le = LabelEncoder()
    le.fit(np.unique(np.concatenate([y_train_labels, y_test_labels])))
    y_train = le.transform(y_train_labels)
    y_test = le.transform(y_test_labels)

    train_counts = np.bincount(y_train, minlength=len(le.classes_))
    test_counts = np.bincount(y_test, minlength=len(le.classes_))
    print("\nWindows per class (TRAIN):")
    for i, cls in enumerate(le.classes_):
        print(f"  {cls}: {train_counts[i]}")
    print("\nWindows per class (VAL):")
    for i, cls in enumerate(le.classes_):
        print(f"  {cls}: {test_counts[i]}")

    feature_names = get_feature_names(preset=args.feature_preset, n_channels=4)
    if len(feature_names) != X_train.shape[1]:
        feature_names = [f"f{i}" for i in range(X_train.shape[1])]

    drop_patterns = [p.strip().lower() for p in args.drop_features.split(",") if p.strip()]
    if drop_patterns:
        keep_mask = np.array(
            [not any(pat in name.lower() for pat in drop_patterns) for name in feature_names],
            dtype=bool,
        )
        removed = [name for name, keep in zip(feature_names, keep_mask) if not keep]
        if not np.any(keep_mask):
            raise ValueError("All features were removed. Adjust --drop_features.")

        X_train = X_train[:, keep_mask]
        X_test = X_test[:, keep_mask]
        feature_names = [name for name, keep in zip(feature_names, keep_mask) if keep]

        print("Dropped feature patterns:", drop_patterns)
        print(f"Removed {len(removed)} features; kept {len(feature_names)}")
        print("Removed feature names:", removed)

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=None if args.max_features == "none" else args.max_features,
        class_weight=None if args.class_weight == "none" else args.class_weight,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    val_acc = clf.score(X_test, y_test)
    print(f"\nTrain accuracy: {train_acc:.4f}")
    print(f"Val accuracy (for model selection): {val_acc:.4f}")

    joblib.dump(clf, "semg_rf.pkl")
    joblib.dump(le, "label_encoder_rf.pkl")
    print("Model saved to semg_rf.pkl")

    label_ids = np.array(le.transform(list(le.classes_))).tolist()
    print("Label mapping:", {cls: lbl for cls, lbl in zip(list(le.classes_), label_ids)})

    preds = clf.predict(X_test)
    pred_names = le.inverse_transform(preds)
    label_names = le.inverse_transform(y_test)
    cm = confusion_matrix(label_names, pred_names, labels=le.classes_)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)

    print("\n" + "=" * 60)
    print("VALIDATION (for model selection - NOT ground-truth)")
    print("=" * 60)
    print(f"Val accuracy: {val_acc:.4f}")
    print("\nConfusion Matrix (Rows: Actual, Columns: Predicted):")
    print(cm_df)
    cm_df.to_csv("confusion_matrix_rf_val.csv")
    print("Saved to: confusion_matrix_rf_val.csv")

    print("\nClassification Report:")
    print(classification_report(label_names, pred_names, target_names=le.classes_))

    feature_imp_df = top_feature_importance(clf, feature_names, top_k=args.top_k_diagnostics)
    feature_imp_df.to_csv("rf_feature_importance_val.csv", index=False)
    print("\nTop feature importances:")
    print(feature_imp_df)
    print("Saved to: rf_feature_importance_val.csv")

    top_confusions_df = top_confusions(cm_df, top_k=args.top_k_diagnostics)
    top_confusions_df.to_csv("rf_top_confusions_val.csv", index=False)
    print("\nMost frequent misclassification pairs:")
    print(top_confusions_df)
    print("Saved to: rf_top_confusions_val.csv")

    bad_leaves_df, branch_feature_df = analyze_bad_leaves(
        clf,
        X_test,
        y_test,
        preds,
        feature_names,
        top_k=args.top_k_diagnostics,
        min_support=args.min_leaf_support,
    )
    bad_leaves_df.to_csv("rf_bad_leaves_val.csv", index=False)
    branch_feature_df.to_csv("rf_branch_feature_pressure_val.csv", index=False)
    print("\nHighest-error RF leaves (candidate bad branches):")
    print(bad_leaves_df)
    print("Saved to: rf_bad_leaves_val.csv")
    print("\nFeatures most used on misclassified decision paths:")
    print(branch_feature_df)
    print("Saved to: rf_branch_feature_pressure_val.csv")

    if args.unseen:
        csv_files = sorted(glob.glob(os.path.join(args.unseen, "*.csv")))
        if not csv_files:
            print(f"\nNo CSV files found in {args.unseen}")
        else:
            all_unseen_preds = []
            all_unseen_labels = []

            for csv_path in csv_files:
                fname = os.path.basename(csv_path)
                df_unseen = pd.read_csv(csv_path)
                try:
                    X_unseen, y_unseen = create_samples(
                        df_unseen, WINDOW_SIZE, step_size, feature_preset=args.feature_preset
                    )
                except ValueError as e:
                    print(f"  Skipping {fname}: {e}")
                    continue

                if drop_patterns:
                    X_unseen = X_unseen[:, keep_mask]

                y_unseen_encoded = le.transform(y_unseen)
                unseen_preds = clf.predict(X_unseen)
                all_unseen_preds.extend(unseen_preds)
                all_unseen_labels.extend(y_unseen_encoded)

            if all_unseen_preds:
                print("\n" + "=" * 60)
                print("UNSEEN DATA - GROUND-TRUTH EVALUATION")
                print("=" * 60)

                agg_preds = np.array(all_unseen_preds)
                agg_labels = np.array(all_unseen_labels)
                agg_acc = np.mean(agg_preds == agg_labels)
                print(f"\nFiles evaluated: {len(csv_files)}")
                print(f"Overall unseen accuracy: {agg_acc:.4f}")

                agg_pred_names = le.inverse_transform(agg_preds)
                agg_label_names = le.inverse_transform(agg_labels)
                agg_cm = confusion_matrix(agg_label_names, agg_pred_names, labels=le.classes_)
                agg_cm_df = pd.DataFrame(agg_cm, index=le.classes_, columns=le.classes_)

                print("\nConfusion Matrix:")
                print("(Rows: Actual, Columns: Predicted)")
                print(agg_cm_df)
                agg_cm_df.to_csv("confusion_matrix_rf_unseen.csv")
                print("Saved to: confusion_matrix_rf_unseen.csv")

                print("\nClassification Report:")
                print(classification_report(agg_label_names, agg_pred_names, target_names=le.classes_))
