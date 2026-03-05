"""Train a Random Forest classifier on sEMG feature data."""

import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from semg_model import extract_features, WINDOW_SIZE, FEATURE_PRESETS

import joblib
import os
import argparse

if os.path.exists("semg_rf.pkl"):
    os.remove("semg_rf.pkl")
    print("Deleted existing semg_rf.pkl")

# ── Args ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data", default="combined.csv")
parser.add_argument(
    "--feature_preset",
    default="baseline",
    choices=list(FEATURE_PRESETS.keys()),
)
parser.add_argument("--unseen", default=None, help="Path to unseen CSV data for evaluation after training")
parser.add_argument("--n_estimators", type=int, default=200)
parser.add_argument("--max_depth", type=int, default=None)
args = parser.parse_args()

# ── Load & window data ───────────────────────────────────────────────
df = pd.read_csv(args.data)
STEP_SIZE = int(WINDOW_SIZE //4 )


def create_samples(df, window_size, step=STEP_SIZE, feature_preset="baseline"):
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


df_train, df_val = time_based_split(df, val_ratio=0.2)
print(f"Time-based split: {len(df_train)} train rows, {len(df_val)} val rows (no leakage)")

X_train, y_train_labels = create_samples(df_train, WINDOW_SIZE, feature_preset=args.feature_preset)
X_test, y_test_labels = create_samples(df_val, WINDOW_SIZE, feature_preset=args.feature_preset)

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

# ── Train ─────────────────────────────────────────────────────────────
clf = RandomForestClassifier(
    n_estimators=args.n_estimators,
    max_depth=args.max_depth,
    random_state=42,
    n_jobs=-1,
)
clf.fit(X_train, y_train)

train_acc = clf.score(X_train, y_train)
val_acc = clf.score(X_test, y_test)
print(f"\nTrain accuracy: {train_acc:.4f}")
print(f"Val accuracy (for model selection): {val_acc:.4f}")

# ── Save ──────────────────────────────────────────────────────────────
joblib.dump(clf, "semg_rf.pkl")
joblib.dump(le, "label_encoder_rf.pkl")
print("Model saved to semg_rf.pkl")

# ── Label mapping ────────────────────────────────────────────────────
label_ids = np.array(le.transform(list(le.classes_))).tolist()
print("Label mapping:", {cls: lbl for cls, lbl in zip(list(le.classes_), label_ids)})

# ── Confusion matrix ─────────────────────────────────────────────────
preds = clf.predict(X_test)
pred_names = le.inverse_transform(preds)
label_names = le.inverse_transform(y_test)
cm = confusion_matrix(label_names, pred_names, labels=le.classes_)
cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)

print("\n" + "=" * 60)
print("VALIDATION (for model selection — NOT ground-truth)")
print("=" * 60)
print(f"Val accuracy: {val_acc:.4f}")
print("\nConfusion Matrix (Rows: Actual, Columns: Predicted):")
print(cm_df)
cm_df.to_csv("confusion_matrix_rf_val.csv")
print("Saved to: confusion_matrix_rf_val.csv")

print("\nClassification Report:")
print(classification_report(label_names, pred_names, target_names=le.classes_))

# ── Unseen data evaluation (folder of CSVs) ──────────────────────────
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
                    df_unseen, WINDOW_SIZE, feature_preset=args.feature_preset
                )
            except ValueError as e:
                print(f"  Skipping {fname}: {e}")
                continue

            y_unseen_encoded = le.transform(y_unseen)
            unseen_preds = clf.predict(X_unseen)
            all_unseen_preds.extend(unseen_preds)
            all_unseen_labels.extend(y_unseen_encoded)

        # ── Overall unseen results ────────────────────────────────────
        if all_unseen_preds:
            print("\n" + "=" * 60)
            print("UNSEEN DATA — GROUND-TRUTH EVALUATION")
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
