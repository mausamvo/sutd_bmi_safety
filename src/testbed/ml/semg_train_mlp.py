import pandas as pd
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from semg_model import SEMGMLP, extract_features, WINDOW_SIZE, SEMGDataset
import joblib
import os
import argparse

BATCH_SIZE = 10
EPOCHS = 100
LEARNING_RATE = 0.001

def parse_feature_list(feature_str):
    feats = [x.strip().lower() for x in feature_str.split(",") if x.strip()]
    if not feats:
        raise ValueError("Feature list is empty.")
    return feats

# 1. Load and preprocess data
# df = pd.read_csv('combined.csv')
if os.path.exists('semg_mlp.pth'):
    print("Model already exists. Exiting to avoid overwriting.")
    exit(0)

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="combined.csv")
# Default features: MAV, RMS, VAR, ZC, SSC (removed WL due to poor performance)
parser.add_argument("--features", default="mav,rms,var,zc,ssc",
                    help="Comma-separated feature list, e.g. mav,rms,var,zc,ssc")
args = parser.parse_args()

feature_names = parse_feature_list(args.features)
print("Using features:", feature_names)

df = pd.read_csv(args.data)

# Group every 100 rows into a sample
def create_samples(df, window_size):
    X, y = [], []
    for i in range(0, len(df), window_size):
        chunk = df.iloc[i:i+window_size]
        if len(chunk) < window_size:
            continue
        sample = chunk[[
            'Ch0 Act',
            'Ch1 Act',
            'Ch2 Act',
            'Ch3 Act',
            ]].values.T  # shape: (4, window_size)
        feats = extract_features(sample, feature_names=feature_names)
        label = chunk['Action'].iloc[0]
        X.append(feats)
        y.append(label)
    return np.stack(X), np.array(y)

X, y = create_samples(df, WINDOW_SIZE)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

train_ds = SEMGDataset(X_train, y_train)
test_ds = SEMGDataset(X_test, y_test)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

input_dim = X_train.shape[1]
n_classes = len(le.classes_)
model = SEMGMLP(n_classes, input_dim)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training loop with early stopping
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_acc = 0.0
best_model_state = None
patience = 15
epochs_no_improve = 0

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
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} - Val Acc: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        best_model_state = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
    if epochs_no_improve >= patience:
        print(f"Early stopping at epoch {epoch+1} with best accuracy: {best_acc:.2%}")
        break

if best_model_state is not None:
    torch.save(best_model_state, 'semg_mlp.pth')
    joblib.dump(le, 'label_encoder_mlp.pkl')
    joblib.dump({"feature_names": feature_names}, "feature_config.pkl")
    print(f"Best model saved with accuracy: {best_acc:.2%}")
else:
    print("No model was saved.")

label_ids = np.array(le.transform(list(le.classes_))).tolist()
label_names = list(le.classes_)
print("Label mapping:")
for name, idx in zip(label_names, label_ids):
    print(f"{name}: {idx}")
