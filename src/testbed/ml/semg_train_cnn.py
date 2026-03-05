import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from semg_model import SEMGCNN, SEMGDataset, WINDOW_SIZE
import joblib

BATCH_SIZE = 10
EPOCHS = 50
LEARNING_RATE = 0.001

# Load and preprocess data
df = pd.read_csv("..\sutd_bmi_safety_data\combined.csv ")


def create_samples(df, window_size):
    X, y = [], []
    for i in range(0, len(df), window_size):
        chunk = df.iloc[i : i + window_size]
        if len(chunk) < window_size:
            continue
        # Use 2 channels: Ch0 Act, Ch1 Act
        sample = chunk[
            ["Ch0 Act", "Ch1 Act", "Ch2 Act", "Ch3 Act"]
        ].values.T  # shape: (2, window_size)
        X.append(sample)
        y.append(chunk["Action"].iloc[0])
    return np.stack(X), np.array(y)


X, y = create_samples(df, WINDOW_SIZE)

# Normalize per channel
X = (X - X.mean(axis=2, keepdims=True)) / (X.std(axis=2, keepdims=True) + 1e-6)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
train_ds = SEMGDataset(X_train, y_train)
test_ds = SEMGDataset(X_test, y_test)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SEMGCNN(n_classes=len(le.classes_)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
best_acc, best_model_state = 0.0, None
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
            preds = model(Xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    acc = correct / total
    if acc > best_acc:
        best_acc = acc
        best_model_state = model.state_dict()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} - Val Acc: {acc:.4f}")

# Save model and encoder
if best_model_state:
    torch.save(best_model_state, "semg_cnn.pth")
    joblib.dump(le, "label_encoder_cnn.pkl")
    print(f"Best model saved with accuracy: {best_acc:.2%}")
