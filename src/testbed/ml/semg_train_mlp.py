import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from semg_model import SEMGMLP, extract_features, WINDOW_SIZE, SEMGDataset
import joblib
import os
import argparse

BATCH_SIZE = 10
EPOCHS = 100
LEARNING_RATE = 0.001

if os.path.exists('semg_mlp.pth'):
    print("Model already exists. Exiting to avoid overwriting.")
    exit(0)
    
# 1. Load and preprocess data
# df = pd.read_csv('combined.csv')
parser = argparse.ArgumentParser()
parser.add_argument("--data", default="combined.csv")
args = parser.parse_args()

df = pd.read_csv(args.data)

# Group every 600 rows as one sample
def create_samples(df, window_size):
    X, y = [], []
    for i in range(0, len(df), window_size):
        chunk = df.iloc[i:i+window_size]
        if len(chunk) < window_size:
            continue
        # Use Ch0 Act and Ch1 Act as 2 channels
        sample = chunk[[
            'Ch0 Act', 
            # 'Ch0 Env', 
            'Ch1 Act', 
            # 'Ch1 Env',
            'Ch2 Act', 
            # 'Ch2 Env', 
            'Ch3 Act', 
            # 'Ch3 Env',
            ]].values.T  # shape: (2, window_size)
        feats = extract_features(sample)
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
    print(f"Best model saved with accuracy: {best_acc:.2%}")
else:
    print("No model was saved.")

# Print label mapping
import numpy as np
label_ids = np.array(le.transform(list(le.classes_))).tolist()
print("Label mapping:", {cls: lbl for cls, lbl in zip(list(le.classes_), label_ids)})


