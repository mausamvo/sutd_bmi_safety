import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from semg_model import SEMGMLP, extract_features, WINDOW_SIZE, SEMGDataset, FEATURE_PRESETS

import joblib
import os
import argparse

BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001

if os.path.exists('semg_mlp.pth'):
    os.remove('semg_mlp.pth')
    print("Deleted existing semg_mlp.pth")
    
# 1. Load and preprocess data
# df = pd.read_csv('combined.csv')
parser = argparse.ArgumentParser()
parser.add_argument("--data", default="combined.csv")
parser.add_argument(
    "--feature_preset",
    default="baseline",
    choices=list(FEATURE_PRESETS.keys())
)
parser.add_argument("--unseen", default=None, help="Folder of unseen CSVs for ground-truth evaluation")
args = parser.parse_args()

df = pd.read_csv(args.data)
# Group rows into overlapping windows
WINDOW_SIZE = int(WINDOW_SIZE)
# STEP_SIZE = int(WINDOW_SIZE)   # no overlap
STEP_SIZE = int(WINDOW_SIZE // 4)  # use this later if you want 75% overlap
def create_samples(df, window_size, step=STEP_SIZE, feature_preset="baseline"):
    window_size = int(window_size)
    step = int(step)

    X, y = [], []
    for i in range(0, len(df) - window_size + 1, step):
        chunk = df.iloc[i:i+window_size]

        label = chunk['Action'].iloc[0]
        if (chunk['Action'] != label).any():
            continue

        sample = chunk[['Ch0 Act', 'Ch1 Act', 'Ch2 Act', 'Ch3 Act']].to_numpy().T
        if sample.shape != (4, window_size):
            continue

        feats = extract_features(sample, preset=feature_preset)
        X.append(feats)
        y.append(label)

    if len(X) == 0:
        raise ValueError("No valid windows created. Check labels / window size / data.")

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

# Z-score standardization (fit on train only, apply to val)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

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

train_ds = SEMGDataset(X_train, y_train)
test_ds = SEMGDataset(X_test, y_test)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

input_dim = X_train.shape[1]
n_classes = len(le.classes_)
model = SEMGMLP(n_classes, input_dim)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 3. Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_acc = 0.0
best_model_state = None
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(Xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * Xb.size(0)
    avg_loss = total_loss / len(train_ds)
    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            out = model(Xb)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    acc = correct / total
    if acc > best_acc:
        best_acc = acc
        best_model_state = model.state_dict()
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} - Val Acc: {acc:.4f}")

# Save best model and label encoder
if best_model_state is not None:
    torch.save(best_model_state, 'semg_mlp.pth')
    joblib.dump(le, 'label_encoder_mlp.pkl')
    joblib.dump(scaler, 'scaler_mlp.pkl')
    print(f"Best model saved with accuracy: {best_acc:.2%}")
else:
    print("No model was saved.")

# Print label mapping
label_ids = np.array(le.transform(list(le.classes_))).tolist()
print("Label mapping:", {cls: lbl for cls, lbl in zip(list(le.classes_), label_ids)})

# ── Validation confusion matrix (for model selection only) ───────────
model.load_state_dict(torch.load('semg_mlp.pth', map_location=device))
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        Xb = Xb.to(device)
        out = model(Xb)
        all_preds.extend(out.argmax(dim=1).cpu().numpy())
        all_labels.extend(yb.numpy())

pred_names = le.inverse_transform(all_preds)
label_names = le.inverse_transform(all_labels)
cm = confusion_matrix(label_names, pred_names, labels=le.classes_)
cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)

print("\n" + "=" * 60)
print("VALIDATION (for model selection — NOT ground-truth)")
print("=" * 60)
val_acc = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"Val accuracy: {val_acc:.4f}")
print("\nConfusion Matrix (Rows: Actual, Columns: Predicted):")
print(cm_df)
cm_df.to_csv("confusion_matrix_mlp_val.csv")
print("Saved to: confusion_matrix_mlp_val.csv")

print("\nClassification Report:")
print(classification_report(label_names, pred_names, target_names=le.classes_))

# ── Unseen data evaluation (folder of CSVs) — GROUND-TRUTH ──────────
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
            X_unseen = scaler.transform(X_unseen)
            unseen_ds = SEMGDataset(X_unseen, y_unseen_encoded)
            unseen_loader = DataLoader(unseen_ds, batch_size=BATCH_SIZE)

            model.eval()
            with torch.no_grad():
                for Xb, yb in unseen_loader:
                    Xb = Xb.to(device)
                    out = model(Xb)
                    all_unseen_preds.extend(out.argmax(dim=1).cpu().numpy())
                    all_unseen_labels.extend(yb.numpy())

        # ── Aggregate across all unseen files ─────────────────────────
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

            print("\nConfusion Matrix (Rows: Actual, Columns: Predicted):")
            print(agg_cm_df)
            agg_cm_df.to_csv("confusion_matrix_mlp_unseen.csv")
            print("Saved to: confusion_matrix_mlp_unseen.csv")

            print("\nClassification Report:")
            print(classification_report(
                agg_label_names, agg_pred_names,
                labels=le.classes_, target_names=le.classes_, zero_division=0
            ))

