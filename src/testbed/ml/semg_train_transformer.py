"""
Train a Transformer (BERT-style) classifier on raw sEMG window data.

Unlike the sklearn models that operate on hand-crafted features, this model
uses a small Transformer encoder that learns directly from the raw
4-channel × 100-timestep windows — similar to how BERT processes sequences.

Architecture:
  - Linear projection of each timestep (4 channels → d_model)
  - Learnable positional embeddings
  - N Transformer encoder layers
  - Mean-pool over time → classification head
"""

import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from semg_model import WINDOW_SIZE

import joblib
import os
import argparse

# ══════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════
class SEMGTransformer(nn.Module):
    """Small BERT-style transformer for sEMG classification."""

    def __init__(
        self,
        n_classes: int,
        n_channels: int = 4,
        seq_len: int = WINDOW_SIZE,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        # Project each timestep (n_channels dims) into d_model dims
        self.input_proj = nn.Linear(n_channels, d_model)

        # Learnable positional embeddings (like BERT)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # CLS token (like BERT)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x):
        # x: (B, 4, T) → (B, T, 4)
        x = x.permute(0, 2, 1)
        B, T, C = x.shape

        x = self.input_proj(x)  # (B, T, d_model)
        x = x + self.pos_embedding[:, :T, :]

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, T+1, d_model)

        x = self.transformer(x)
        x = self.norm(x)

        # Use CLS token output for classification
        cls_out = x[:, 0, :]
        return self.classifier(cls_out)


class RawSEMGDataset(Dataset):
    """Dataset that returns raw (4, window_size) arrays."""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    if os.path.exists("semg_transformer.pth"):
        os.remove("semg_transformer.pth")
        print("Deleted existing semg_transformer.pth")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="combined.csv")
    parser.add_argument("--unseen", default=None, help="Path to unseen CSV data for evaluation after training")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    args = parser.parse_args()

    # ── Load & window (raw, no hand-crafted features) ────────────────
    df = pd.read_csv(args.data)
    window_size = int(WINDOW_SIZE)
    step = window_size // 4

    def create_raw_samples(df, window_size, step):
        X, y = [], []
        for i in range(0, len(df) - window_size + 1, step):
            chunk = df.iloc[i : i + window_size]
            label = chunk["Action"].iloc[0]
            if (chunk["Action"] != label).any():
                continue
            sample = chunk[["Ch0 Act", "Ch1 Act", "Ch2 Act", "Ch3 Act"]].to_numpy().T
            if sample.shape != (4, window_size):
                continue
            X.append(sample)
            y.append(label)
        if len(X) == 0:
            raise ValueError("No valid windows created.")
        return np.stack(X), np.array(y)

    # ── Time-based split (no leakage) ────────────────────────────────
    def time_based_split(df, val_ratio=0.2):
        """Split each contiguous action segment chronologically."""
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

    X_train, y_train_labels = create_raw_samples(df_train, window_size, step)
    X_test, y_test_labels = create_raw_samples(df_val, window_size, step)
    print(f"Raw sample shape: {X_train.shape}  (samples, channels, timesteps)")

    le = LabelEncoder()
    le.fit(np.unique(np.concatenate([y_train_labels, y_test_labels])))
    y_train = le.transform(y_train_labels)
    y_test = le.transform(y_test_labels)
    n_classes = len(le.classes_)

    train_counts = np.bincount(y_train, minlength=n_classes)
    test_counts = np.bincount(y_test, minlength=n_classes)
    print("\nWindows per class (TRAIN):")
    for i, cls in enumerate(le.classes_):
        print(f"  {cls}: {train_counts[i]}")
    print("\nWindows per class (VAL):")
    for i, cls in enumerate(le.classes_):
        print(f"  {cls}: {test_counts[i]}")

    train_ds = RawSEMGDataset(X_train, y_train)
    test_ds = RawSEMGDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # ── Model ────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SEMGTransformer(
        n_classes=n_classes,
        n_channels=4,
        seq_len=window_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training ─────────────────────────────────────────────────────
    best_acc = 0.0
    best_model_state = None

    for epoch in range(args.epochs):
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
        scheduler.step()

        # Validate
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
        print(
            f"Epoch {epoch+1}/{args.epochs} - "
            f"Loss: {avg_loss:.4f} - Val Acc: {acc:.4f} - "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

    # ── Save ─────────────────────────────────────────────────────────
    if best_model_state is not None:
        torch.save(best_model_state, "semg_transformer.pth")
        joblib.dump(le, "label_encoder_transformer.pkl")
        print(f"\nBest model saved with accuracy: {best_acc:.2%}")
    else:
        print("No model was saved.")

    # ── Label mapping ────────────────────────────────────────────────
    label_ids = np.array(le.transform(list(le.classes_))).tolist()
    print(
        "Label mapping:",
        {cls: lbl for cls, lbl in zip(list(le.classes_), label_ids)},
    )

    # ── Confusion matrix ─────────────────────────────────────────────
    model.load_state_dict(torch.load("semg_transformer.pth", map_location=device))
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
    val_acc_final = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"Val accuracy: {val_acc_final:.4f}")
    print("\nConfusion Matrix (Rows: Actual, Columns: Predicted):")
    print(cm_df)
    cm_df.to_csv("confusion_matrix_transformer_val.csv")
    print("Saved to: confusion_matrix_transformer_val.csv")

    print("\nClassification Report:")
    print(classification_report(label_names, pred_names, target_names=le.classes_))

    # ── Unseen data evaluation (folder of CSVs) ──────────────────────
    if args.unseen:
        import glob
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
                    X_unseen, y_unseen = create_raw_samples(df_unseen, window_size, step)
                except ValueError as e:
                    print(f"  Skipping {fname}: {e}")
                    continue

                y_unseen_encoded = le.transform(y_unseen)
                unseen_ds = RawSEMGDataset(X_unseen, y_unseen_encoded)
                unseen_loader = DataLoader(unseen_ds, batch_size=args.batch_size)

                model.eval()
                with torch.no_grad():
                    for Xb, yb in unseen_loader:
                        Xb = Xb.to(device)
                        out = model(Xb)
                        all_unseen_preds.extend(out.argmax(dim=1).cpu().numpy())
                        all_unseen_labels.extend(yb.numpy())

            # ── Aggregate across all unseen files ─────────────────────
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
                agg_cm_df.to_csv("confusion_matrix_transformer_unseen.csv")
                print("Saved to: confusion_matrix_transformer_unseen.csv")

                print("\nClassification Report:")
                print(classification_report(agg_label_names, agg_pred_names, target_names=le.classes_))
