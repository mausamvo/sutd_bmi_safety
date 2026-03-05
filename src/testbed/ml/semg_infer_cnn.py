You said:
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
# Group rows into overlapping windows
STEP_SIZE = WINDOW_SIZE 

def create_samples(df, window_size, step=STEP_SIZE):
    X, y = [], []
    for i in range(0, len(df) - window_size + 1, step):
        chunk = df.iloc[i:i+window_size]
        # Only use window if all rows have the same action label
        label = chunk['Action'].iloc[0]
        if (chunk['Action'] != label).any():
            continue
        sample = chunk[[
            'Ch0 Act', 
            # 'Ch0 Env', 
            'Ch1 Act', 
            # 'Ch1 Env',
            'Ch2 Act', 
            # 'Ch2 Env', 
            'Ch3 Act', 
            # 'Ch3 Env',
            ]].values.T  # shape: (4, window_size)
        feats = extract_features(sample)
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

# Confusion matrix on validation set
from sklearn.metrics import confusion_matrix
import pandas as pd

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

print("\nValidation Confusion Matrix:")
print("(Rows: Actual, Columns: Predicted)")
print(cm_df)

cm_df.to_csv("confusion_matrix_train_val.csv")
print("\nSaved to: confusion_matrix_train_val.csv")


ration 13/20
============================================================

Training failed:
Traceback (most recent call last):
  File "C:\Users\leewe\Desktop\semg\sutd_bmi_safety\src\testbed\ml\semg_train_mlp.py", line 55, in <module>
    X, y = create_samples(df, WINDOW_SIZE)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\leewe\Desktop\semg\sutd_bmi_safety\src\testbed\ml\semg_train_mlp.py", line 34, in create_samples
    for i in range(0, len(df) - window_size + 1, step):
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: 'float' object cannot be interpreted as an integer


============================================================
Iteration 14/20
============================================================
Traceback (most recent call last):
  File "C:\Users\leewe\Desktop\semg\sutd_bmi_safety\src\testbed\ml\semg_search_best.py", line 34, in <module>
    train_result = subprocess.run(
                   ^^^^^^^^^^^^^^^
  File "C:\Users\leewe\AppData\Local\Programs\Python\Python312\Lib\subprocess.py", line 550, in run
    stdout, stderr = process.communicate(input, timeout=timeout)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\leewe\AppData\Local\Programs\Python\Python312\Lib\subprocess.py", line 1209, in communicate
    stdout, stderr = self._communicate(input, endtime, timeout)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\leewe\AppData\Local\Programs\Python\Python312\Lib\subprocess.py", line 1628, in _communicate
    self.stdout_thread.join(self._remaining_time(endtime))
  File "C:\Users\leewe\AppData\Local\Programs\Python\Python312\Lib\threading.py", line 1147, in join
    self._wait_for_tstate_lock()
  File "C:\Users\leewe\AppData\Local\Programs\Python\Python312\Lib\threading.py", line 1167, in _wait_for_tstate_lock
    if lock.acquire(block, timeout):
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ChatGPT said:

That error is coming from range(...) needing integers, but your window_size or step is a float.

In your code, both come from:

    WINDOW_SIZE (imported from semg_model)

    STEP_SIZE = WINDOW_SIZE

    create_samples(df, WINDOW_SIZE) → for i in range(..., step):

So if WINDOW_SIZE is 200.0 (float) instead of 200 (int), you’ll get:

    TypeError: 'float' object cannot be interpreted as an integer

Fix (do this in your training script)

Force both to int right before you use them:

# After importing WINDOW_SIZE
WINDOW_SIZE = int(WINDOW_SIZE)

STEP_SIZE = WINDOW_SIZE  # now int

def create_samples(df, window_size, step=STEP_SIZE):
    window_size = int(window_size)
    step = int(step)
    X, y = [], []
    for i in range(0, len(df) - window_size + 1, step):
        ...

That alone should fix Iteration 13/20.
Better fix (do this in semg_model too)

Wherever WINDOW_SIZE is defined in semg_model.py, make sure it’s an int:

WINDOW_SIZE = 200   # not 200.0
# OR if it’s computed:
WINDOW_SIZE = int(WINDOW_SIZE)

Why your Iteration 14/20 looks different

The second traceback is from your semg_search_best.py running training via subprocess.run(...) and then hanging / timing out while waiting for output.

That’s happening because:

    training crashed in Iteration 13, and/or

    your subprocess call is capturing output and something is blocking.

Once you fix the float→int issue, this often goes away. If it still hangs, tell me how you call subprocess.run (the exact lines around it), and I’ll point out the safe pattern (usually text=True, not huge captured output, and avoid waiting forever).
Quick sanity check

Add this print once before create_samples:

print("WINDOW_SIZE:", WINDOW_SIZE, type(WINDOW_SIZE))
print("STEP_SIZE:", STEP_SIZE, type(STEP_SIZE))

You should see <class 'int'>.

You said:
import os
import numpy as np
import torch
from collections import deque
import socketio
import time
from semg_model import extract_features, SEMGMLP, WINDOW_SIZE
import joblib
import argparse
import pandas as pd
import sys
from sklearn.metrics import confusion_matrix
import datetime

# Load label encoder and model
le = joblib.load("label_encoder_mlp.pkl")
# Model must be created with correct shape and class count
input_dim = 4 * 6  # 4 channels x 6 features
n_classes = len(le.classes_)
model = SEMGMLP(n_classes, input_dim)
model.load_state_dict(torch.load("semg_mlp.pth", map_location="cpu"))
model.eval()

# Initialize buffer
buffer = deque(maxlen=WINDOW_SIZE)


def process_and_predict(data):
    try:
        raw = np.array(
            [
                data["ch0"]["a"],
                # data["ch0"]["e"],
                data["ch1"]["a"],
                # data["ch1"]["e"],
                data["ch2"]["a"],
                # data["ch2"]["e"],
                data["ch3"]["a"],
                # data["ch3"]["e"],
            ]
        )
        buffer.append(raw)
        if len(buffer) == WINDOW_SIZE:
            window = np.stack(buffer, axis=1)
            feats = extract_features(window)
            arr = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                out = model(arr)
                pred = out.argmax(dim=1).item()
                buffer.clear()
                return le.inverse_transform([pred])[0]
    except Exception as e:
        print(f"Inference error: {e}")
    return None


sio = socketio.Client()


@sio.event
def connect():
    print("Connected to Socket.IO server.")


@sio.on("adc_data")
def semg_data(data):
    prediction = process_and_predict(data)
    if prediction:
        print(f"Detection: {prediction}")


def offline_predict(csv_path):
    if "*" in csv_path:
        import glob
        csv_files = glob.glob(csv_path)
        if not csv_files:
            print("No CSV files found matching the pattern.")
            return
    else:
        csv_files = [csv_path]

    correct = 0

    # Initialize counters
    TP = {cls: 0 for cls in le.classes_}
    FP = {cls: 0 for cls in le.classes_}
    Total_per_class = {cls: 0 for cls in le.classes_}
    total_predictions = 0
    
    # Store all predictions for confusion matrix and CSV output
    all_actuals = []
    all_predictions = []
    prediction_records = []

    for csv_path in csv_files:
        buffer.clear()
        basename = os.path.basename(csv_path)
        ground_truth = None
        for cls in le.classes_:
            if cls in basename:
                ground_truth = cls
                break

        # print(f"Processing offline data from {basename}")
        df = pd.read_csv(csv_path)
        # Expecting columns: ch0_a, ch1_a, ch2_a, ch3_a
        for i in range(len(df)):
            data = {
                "ch0": {"a": df.iloc[i]["Ch0 Act"]},
                "ch1": {"a": df.iloc[i]["Ch1 Act"]},
                "ch2": {"a": df.iloc[i]["Ch2 Act"]},
                "ch3": {"a": df.iloc[i]["Ch3 Act"]},
            }
            pred = process_and_predict(data)
            if pred:
                total_predictions += 1
                # print(f"Offline Detection at row {i}: {pred}")
                
                # Record prediction
                all_actuals.append(ground_truth)
                all_predictions.append(pred)
                prediction_records.append({
                    'file': basename,
                    'actual': ground_truth,
                    'predicted': pred
                })
                
                if pred == ground_truth:
                    correct += 1
                    TP[ground_truth] += 1
                else:
                    FP[pred] += 1
                
                Total_per_class[ground_truth] += 1
        buffer.clear()

    # Accuracy
    accuracy = correct / total_predictions if total_predictions > 0 else 0.0

    # Per-class metrics
    print("\nMetrics per class:")
    for cls in le.classes_:
        class_accuracy = TP[cls]  / Total_per_class[cls] if Total_per_class[cls] > 0 else 0.0

        print(f"Class: {cls}")
        print(f"  Accuracy:  {class_accuracy:.4f}")

    print("\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}\n")
    
    # Save predictions to CSV
    if prediction_records:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"predictions_{timestamp}.csv"
        results_df = pd.DataFrame(prediction_records)
        results_df.to_csv(results_filename, index=False)
        print(f"Predictions saved to: {results_filename}")
    
    # Display confusion matrix
    if all_actuals and all_predictions:
        cm = confusion_matrix(all_actuals, all_predictions, labels=le.classes_)
        cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
        
        print("\nConfusion Matrix:")
        print("(Rows: Actual, Columns: Predicted)")
        print(cm_df)
        
        # Save confusion matrix to CSV
        cm_filename = f"confusion_matrix_{timestamp}.csv"
        cm_df.to_csv(cm_filename)
        print(f"\nConfusion matrix saved to: {cm_filename}")
    


if __name__ == "__main__":
    print(" ".join(sys.argv))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["online", "offline"],
        default="online",
        help="Detection mode",
    )
    parser.add_argument("--csv", type=str, help="CSV file for offline mode")
    args = parser.parse_args()

    if args.mode == "online":
        sio.connect("http://localhost:3000", transports=["websocket"], namespaces=["/"])
        try:
            print("waiting...")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            sio.disconnect()
    elif args.mode == "offline":
        if not args.csv:
            print("Please provide a CSV file with --csv for offline mode.")
        else:
            offline_predict(args.csv)

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
# Group rows into overlapping windows
STEP_SIZE = int(WINDOW_SIZE /4)  # 75% overlap

def create_samples(df, window_size, step=STEP_SIZE):
    X, y = [], []
    for i in range(0, len(df) - window_size + 1, step):
        chunk = df.iloc[i:i+window_size]
        # Only use window if all rows have the same action label
        label = chunk['Action'].iloc[0]
        if (chunk['Action'] != label).any():
            continue
        sample = chunk[[
            'Ch0 Act', 
            # 'Ch0 Env', 
            'Ch1 Act', 
            # 'Ch1 Env',
            'Ch2 Act', 
            # 'Ch2 Env', 
            'Ch3 Act', 
            # 'Ch3 Env',
            ]].values.T  # shape: (4, window_size)
        feats = extract_features(sample)
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

# Confusion matrix on validation set
from sklearn.metrics import confusion_matrix
import pandas as pd

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

print("\nValidation Confusion Matrix:")
print("(Rows: Actual, Columns: Predicted)")
print(cm_df)

cm_df.to_csv("confusion_matrix_train_val.csv")
print("\nSaved to: confusion_matrix_train_val.csv")




how do i keep the windows size consistent 

ChatGPT said:

To keep window size consistent, you need 2 things to match everywhere:

    Training windows: always build samples with exactly WINDOW_SIZE rows

    Inference windows (online + offline): only predict when your buffer has exactly WINDOW_SIZE rows, and extract features from a (4, WINDOW_SIZE) window

Right now you’re almost doing it, but there are a few common ways it accidentally becomes inconsistent.
Do these 5 fixes/patterns
1) Force WINDOW_SIZE to be an int once (training + inference)

In both scripts, right after import:

from semg_model import WINDOW_SIZE
WINDOW_SIZE = int(WINDOW_SIZE)

(If WINDOW_SIZE is ever computed/divided, it can become float.)
2) Make step an int and don’t use it as the default arg

Python evaluates default args once, so it can lock in an old value. Do:

def create_samples(df, window_size, step=None):
    window_size = int(window_size)
    step = int(step if step is not None else window_size // 4)  # 75% overlap
    ...
    for i in range(0, len(df) - window_size + 1, step):
        ...

Call it like:

X, y = create_samples(df, WINDOW_SIZE, step=WINDOW_SIZE//4)

3) Enforce shape before feature extraction

In training:

sample = chunk[['Ch0 Act','Ch1 Act','Ch2 Act','Ch3 Act']].to_numpy().T  # (4, W)
if sample.shape != (4, window_size):
    continue
feats = extract_features(sample)

In inference (online/offline), right before extract_features:

window = np.stack(buffer, axis=1)      # should be (4, W)
if window.shape != (4, WINDOW_SIZE):
    return None
feats = extract_features(window)

This guarantees you never accidentally pass a wrong-length window into extract_features.
4) Decide “overlap vs no overlap” consistently

    Training: you use overlap (step = WINDOW_SIZE//4).

    Online/offline inference: you currently do no overlap because you buffer.clear() after a prediction.

If you want inference to behave like training (more stable + more predictions), use a sliding window:

# instead of buffer.clear()
for _ in range(WINDOW_SIZE // 4):   # shift by step (75% overlap)
    buffer.popleft()

…and define STEP = WINDOW_SIZE // 4 once.

If you want non-overlapping predictions, keep buffer.clear(). Window size is still consistent; it’s just less frequent.
5) Make sure your offline loop fills exactly WINDOW_SIZE samples per prediction

Your process_and_predict already waits for len(buffer) == WINDOW_SIZE, so it’s consistent.

But your offline “ground truth per file” logic assumes file contains only that class. If a CSV file contains mixed actions, your window will still be consistent, but your labels become wrong. (Not a window-size issue, but it will ruin metrics.)
Minimal “gold standard” pattern

Use the same constants in both scripts:

WINDOW_SIZE = int(WINDOW_SIZE)
STEP = WINDOW_SIZE // 4  # if you want overlap
buffer = deque(maxlen=WINDOW_SIZE)

And the same window-shape check before extract_features.

If you paste your extract_features function (or tell me what it expects: per-channel features, flatten order, etc.), I can also confirm that your input_dim = 4 * 6 is always consistent with what extract_features returns.

You said:
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
# Group rows into overlapping windows
STEP_SIZE = int(WINDOW_SIZE)  
def create_samples(df, window_size, step=STEP_SIZE):
    X, y = [], []
    for i in range(0, len(df) - window_size + 1, step):
        chunk = df.iloc[i:i+window_size]
        # Only use window if all rows have the same action label
        label = chunk['Action'].iloc[0]
        if (chunk['Action'] != label).any():
            continue
        sample = chunk[[
            'Ch0 Act', 
            # 'Ch0 Env', 
            'Ch1 Act', 
            # 'Ch1 Env',
            'Ch2 Act', 
            # 'Ch2 Env', 
            'Ch3 Act', 
            # 'Ch3 Env',
            ]].values.T  # shape: (4, window_size)
        feats = extract_features(sample)
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

# Confusion matrix on validation set
from sklearn.metrics import confusion_matrix
import pandas as pd

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

print("\nValidation Confusion Matrix:")
print("(Rows: Actual, Columns: Predicted)")
print(cm_df)

cm_df.to_csv("confusion_matrix_train_val.csv")
print("\nSaved to: confusion_matrix_train_val.csv")



what features should i add to improve the mlp model,the accuracy of unseen is pretty good, except for flexion and grasp,do i just recollect data? 

ChatGPT said:

Flexion + grasp being the weak classes usually means either (A) not enough / not diverse enough examples, or (B) your current features don’t separate them well (they can look similar in EMG), or (C) labels/segmentation are noisy.

You don’t have to jump straight to recollecting data, but you should first do these in order:
1) Check if it’s just “not enough flexion/grasp samples”

Print support per class (how many windows in train/test). If flexion/grasp have way fewer windows, accuracy will suffer even if overall accuracy is high.

Quick check:

    count windows per class in y (after create_samples)

    also check per class in y_test

If flexion/grasp support is small → yes, collect more (or oversample/augment).
2) Add features that separate flexion vs grasp better

Right now your extract_features seems to be 4 channels × 6 features (time-domain). Good start, but EMG often needs more shape + frequency info.

Strong feature additions (still MLP-friendly):
Time-domain (easy, very effective)

Per channel:

    RMS

    MAV (mean absolute value)

    Waveform length (WL) (captures “activity”)

    Zero crossings (ZC) (with a small threshold)

    Slope sign changes (SSC) (with threshold)

    Willison amplitude (WAMP) (counts large jumps)

Flexion vs grasp often differs in activation pattern and “busy-ness” → WL / SSC / WAMP help.
Frequency-domain (big improvement for confusing classes)

Per channel from FFT:

    band power in a few bands (e.g., 20–60, 60–100, 100–200 Hz)

    median frequency

    mean frequency

    spectral entropy (optional but nice)

These can separate “same amplitude, different muscle firing texture”.
Cross-channel features (helps grasp a lot)

    pairwise correlation between channels (how “together” they fire)

    ratio features like RMS_ch0 / (RMS_ch1 + eps) etc.
    Grasp often recruits multiple muscles → correlation/ratios help.

If you have Env channels, use them

You commented out ChX Env. If those are meaningful (even if noisy), try including them or at least using them to normalize Act:

    Act - Env or Act / (Env + eps) depending on what Env represents.

3) Improve training so the model “cares” about flexion/grasp

Even with better features, if the model mostly optimizes for easy classes:

    Class-weighted loss

class_counts = np.bincount(y_train)
weights = (class_counts.sum() / (len(class_counts) * class_counts))
criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device))

    Increase batch size a bit (10 is quite small) → try 32 if your dataset isn’t tiny.

    Add BatchNorm + Dropout in the MLP (usually improves generalization & stability).

4) Fix the windowing/labels for EMG

Your training windows only keep chunks where all rows have the same Action. That’s good, but for real EMG:

    Make sure you’re not training mostly on “easy steady parts” and testing on “transition-ish” parts.

    Consider 75% overlap in training windows (you already tried earlier). More windows = better for rare classes.

5) When recollecting data is the right answer

Recollect if:

    flexion/grasp have low sample counts

    recordings are from only one person/session

    labels are shaky (e.g., “grasp” includes relaxed periods)

    flexion/grasp look similar because the gesture execution varies a lot

When collecting, aim for:

    multiple sessions (different days)

    different intensities (light/medium/strong)

    slight arm rotations
    That diversity matters more than just “more of the same”.

What I’d do first (fastest win)

    Print windows-per-class (train/test)

    Add WL + ZC + SSC + WAMP (still simple)

    Add FFT band powers + median frequency

    Use class-weighted loss

If you paste your current extract_features() (or just list the 6 features you already compute), I can suggest the next 6–10 to add in a clean, consistent way (and update input_dim accordingly).

You said:
Print windows-per-class (train/test)
how do i do this

ChatGPT said:

Add these prints right after your train_test_split(...) so you can see class balance in both sets.

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ---- Print windows per class (train/test) ----
# y_train / y_test are encoded ints, so convert counts back to class names
train_counts = np.bincount(y_train, minlength=len(le.classes_))
test_counts  = np.bincount(y_test,  minlength=len(le.classes_))

print("\nWindows per class (TRAIN):")
for i, cls in enumerate(le.classes_):
    print(f"  {cls}: {train_counts[i]}")

print("\nWindows per class (TEST):")
for i, cls in enumerate(le.classes_):
    print(f"  {cls}: {test_counts[i]}")

# Optional: percentages (useful to spot imbalance)
print("\nClass distribution (%):")
print("TRAIN:")
train_total = len(y_train)
for i, cls in enumerate(le.classes_):
    pct = (train_counts[i] / train_total * 100) if train_total > 0 else 0
    print(f"  {cls}: {pct:.1f}%")

print("TEST:")
test_total = len(y_test)
for i, cls in enumerate(le.classes_):
    pct = (test_counts[i] / test_total * 100) if test_total > 0 else 0
    print(f"  {cls}: {pct:.1f}%")

Extra useful check (before split)

This tells you the total windows generated per class from create_samples, so you know whether flexion/grasp are already underrepresented before splitting:

# After y_encoded = le.fit_transform(y)
all_counts = np.bincount(y_encoded, minlength=len(le.classes_))
print("\nWindows per class (ALL):")
for i, cls in enumerate(le.classes_):
    print(f"  {cls}: {all_counts[i]}")

If you want, I can also give you a tiny snippet to print precision/recall/F1 per class (that helps more than accuracy for weak classes like grasp/flexion).

You said:
Add WL + ZC + SSC + WAMP

what does this do and how to add , while keeping original code in case i want to revert, make it modular and i can choose settings i guess?

ChatGPT said:

Nice move. Those four are classic EMG features and they often help exactly with “confusing gesture” classes like flexion vs grasp.
What they do

Per channel, on one window:

    WL (Waveform Length)
    Sum of absolute changes between consecutive samples.
    Helps capture how “busy” / active the signal is.

    ZC (Zero Crossings)
    Counts how many times the signal crosses zero (with a threshold to ignore tiny noise).
    Helps capture frequency-ish behavior in time domain.

    SSC (Slope Sign Changes)
    Counts how often the slope changes sign (peaks/valleys), also with threshold.
    Helps capture signal complexity.

    WAMP (Willison Amplitude)
    Counts how many consecutive sample differences exceed a threshold.
    Helps capture burst activity / muscle activation intensity transitions.

These are all time-domain EMG features, so they’re easy to add and fast to compute (great for MLP + real-time inference).
Best way to add them (modular + reversible)

Make feature extraction configurable in semg_model.py, so your train/inference scripts don’t need big changes.
1) In semg_model.py, create a configurable feature extractor

Add something like this (you can keep your current extract_features and wrap it):

import numpy as np

# Example feature config presets
FEATURE_PRESETS = {
    "baseline": {
        "use_existing": True,   # your current 6 features/channel
        "add_wl": False,
        "add_zc": False,
        "add_ssc": False,
        "add_wamp": False,
        "zc_thresh": 0.01,
        "ssc_thresh": 0.01,
        "wamp_thresh": 0.02,
    },
    "td_plus": {
        "use_existing": True,
        "add_wl": True,
        "add_zc": True,
        "add_ssc": True,
        "add_wamp": True,
        "zc_thresh": 0.01,
        "ssc_thresh": 0.01,
        "wamp_thresh": 0.02,
    },
    "td_only_new": {
        "use_existing": False,
        "add_wl": True,
        "add_zc": True,
        "add_ssc": True,
        "add_wamp": True,
        "zc_thresh": 0.01,
        "ssc_thresh": 0.01,
        "wamp_thresh": 0.02,
    }
}

# --- New helper functions ---
def feat_wl(x: np.ndarray) -> float:
    # Waveform Length
    return float(np.sum(np.abs(np.diff(x))))

def feat_zc(x: np.ndarray, thresh: float = 0.01) -> float:
    # Zero crossings with threshold to suppress noise
    x1 = x[:-1]
    x2 = x[1:]
    crossings = ((x1 * x2) < 0) & (np.abs(x1 - x2) >= thresh)
    return float(np.sum(crossings))

def feat_ssc(x: np.ndarray, thresh: float = 0.01) -> float:
    # Slope Sign Changes with threshold
    x_prev = x[:-2]
    x_curr = x[1:-1]
    x_next = x[2:]
    s1 = x_curr - x_prev
    s2 = x_curr - x_next
    changes = ((s1 * s2) > 0) & ((np.abs(s1) >= thresh) | (np.abs(s2) >= thresh))
    return float(np.sum(changes))

def feat_wamp(x: np.ndarray, thresh: float = 0.02) -> float:
    # Willison Amplitude
    return float(np.sum(np.abs(np.diff(x)) >= thresh))


# --- Keep your original implementation safe ---
def extract_features_baseline(sample: np.ndarray) -> np.ndarray:
    """
    Your ORIGINAL feature extractor.
    sample shape: (n_channels, window_size)
    Returns flat feature vector.
    """
    # TODO: paste your current original extract_features code here unchanged
    # For now, example placeholder:
    feats = []
    for ch in range(sample.shape[0]):
        x = sample[ch]
        feats.extend([
            np.mean(np.abs(x)),     # MAV (example)
            np.sqrt(np.mean(x**2)), # RMS
            np.var(x),              # VAR
            np.mean(x),             # Mean
            np.max(x),              # Max
            np.min(x),              # Min
        ])
    return np.array(feats, dtype=np.float32)


def extract_features_configurable(sample: np.ndarray, config: dict) -> np.ndarray:
    """
    sample shape: (n_channels, window_size)
    Returns flat feature vector.
    """
    feats = []

    # 1) Existing features (optional)
    if config.get("use_existing", True):
        base = extract_features_baseline(sample)
        feats.extend(base.tolist())

    # 2) Add new EMG time-domain features per channel
    for ch in range(sample.shape[0]):
        x = sample[ch].astype(np.float32)

        if config.get("add_wl", False):
            feats.append(feat_wl(x))
        if config.get("add_zc", False):
            feats.append(feat_zc(x, thresh=config.get("zc_thresh", 0.01)))
        if config.get("add_ssc", False):
            feats.append(feat_ssc(x, thresh=config.get("ssc_thresh", 0.01)))
        if config.get("add_wamp", False):
            feats.append(feat_wamp(x, thresh=config.get("wamp_thresh", 0.02)))

    return np.array(feats, dtype=np.float32)


# Backward-compatible default function name used by your scripts
def extract_features(sample: np.ndarray, preset: str = "baseline", config: dict = None) -> np.ndarray:
    """
    Default entry point. Keeps old scripts working.
    """
    if config is None:
        config = FEATURE_PRESETS[preset]
    return extract_features_configurable(sample, config)

2) Add a feature preset argument to your training script

This lets you switch between old and new features without editing code every time.
Change imports

from semg_model import SEMGMLP, extract_features, WINDOW_SIZE, SEMGDataset, FEATURE_PRESETS

Add CLI arg

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="combined.csv")
parser.add_argument("--feature_preset", default="baseline", choices=["baseline", "td_plus", "td_only_new"])
args = parser.parse_args()

Use it in create_samples

def create_samples(df, window_size, step=None, feature_preset="baseline"):
    window_size = int(window_size)
    step = int(step if step is not None else window_size)
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
        raise ValueError("No valid windows were created. Check WINDOW_SIZE / labels / data.")
    return np.stack(X), np.array(y)

Call it

X, y = create_samples(df, WINDOW_SIZE, step=int(WINDOW_SIZE), feature_preset=args.feature_preset)
print("Feature preset:", args.feature_preset)
print("Feature dimension:", X.shape[1])

3) Use the same preset in inference (important!)

Training and inference must use the same feature set, or the model input size won’t match.

In your inference script, add the same arg:

from semg_model import extract_features, SEMGMLP, WINDOW_SIZE, FEATURE_PRESETS
...
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["online", "offline"], default="online")
parser.add_argument("--csv", type=str)
parser.add_argument("--feature_preset", default="baseline", choices=["baseline", "td_plus", "td_only_new"])
args = parser.parse_args()
...
FEATURE_PRESET = args.feature_preset

Then in process_and_predict:

feats = extract_features(window, preset=FEATURE_PRESET)

And build model input dim dynamically (safer than hardcoding 4*6):

# Build a dummy feature vector to get input_dim
dummy = np.zeros((4, int(WINDOW_SIZE)), dtype=np.float32)
input_dim = extract_features(dummy, preset=FEATURE_PRESET).shape[0]
model = SEMGMLP(n_classes, input_dim)

4) Optional: add thresholds as CLI knobs (advanced but useful)

Different datasets need different thresholds. You can expose them later if needed:

    --zc_thresh

    --ssc_thresh

    --wamp_thresh

But start with defaults first.
5) How to test cleanly (and revert anytime)
Original features (baseline)

python semg_train_mlp.py --data combined.csv --feature_preset baseline

Original + WL/ZC/SSC/WAMP

python semg_train_mlp.py --data combined.csv --feature_preset td_plus

Then evaluate offline with same preset:

python semg_detect.py --mode offline --csv "your_test*.csv" --feature_preset td_plus

If results get worse, you can instantly go back to baseline.
Extra tip for flexion/grasp specifically

If WL/ZC/SSC/WAMP help a little but not enough, next best add-ons are:

    MAV slope (trend across subsegments)

    Autoregressive (AR) coefficients (order 4 is common)

    Band powers (frequency-domain)

But WL/ZC/SSC/WAMP is the right next step first.

If you want, paste your current original extract_features() and I’ll merge it exactly into this modular version so you don’t lose anything.

You said:
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset 

# Parameters
WINDOW_SIZE = 100 # this is roughly 1 seconds

class SEMGCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)

        self.fc1 = nn.Linear(32 * (WINDOW_SIZE // 2), 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        return self.fc2(x)
    
# PyTorch Dataset
class SEMGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SEMGMLP(nn.Module):
    def __init__(self, n_classes, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, n_classes)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def extract_features(window):
    """
    Extract classic sEMG features for each channel in a window.
    window: np.ndarray shape (channels, window_size)
    Returns: np.ndarray shape (channels * num_features,)
    Features: MAV, RMS, WL, VAR, ZC, SSC
    """
    def mav(x): return np.mean(np.abs(x))
    def rms(x): return np.sqrt(np.mean(x ** 2))
    def wl(x): return np.sum(np.abs(np.diff(x)))
    def var(x): return np.var(x)
    def zc(x, threshold=0.01): return np.sum(((x[:-1] * x[1:]) < 0) & (np.abs(x[:-1] - x[1:]) > threshold))
    def ssc(x, threshold=0.01):
        return np.sum(((x[1:-1] - x[:-2]) * (x[1:-1] - x[2:]) > 0) &
                      (np.abs(x[1:-1] - x[:-2]) > threshold) &
                      (np.abs(x[1:-1] - x[2:]) > threshold))
    feats = []
    for ch in window:
        feats.extend([
            mav(ch),
            rms(ch),
            wl(ch),
            var(ch),
            zc(ch),
            ssc(ch),
        ])
    return np.array(feats, dtype=np.float32)
here is how my semg_model.py looks, this is the baseline model

ChatGPT said:

Perfect — your current baseline already includes WL, ZC, SSC 🎯
So the only “new” one from that list is WAMP.

That means your baseline features are already:

    MAV

    RMS

    WL

    VAR

    ZC

    SSC

So to improve flexion/grasp, a good next step is:

    Keep baseline as-is

    Add optional WAMP

    Make feature extraction modular/preset-based

    (Optional later) add more features like AR/bandpower

What WAMP adds

WAMP (Willison Amplitude) counts how many times the change between consecutive samples exceeds a threshold.

It helps capture:

    burstiness / activation transitions

    subtle differences in muscle recruitment patterns

This can help distinguish gestures that have similar average amplitude but different signal dynamics (like grasp vs flexion).
✅ Modular version of semg_model.py (baseline preserved)

You can replace your extract_features part with this version.
It keeps your current baseline untouched and lets you switch presets like "baseline" vs "baseline_plus_wamp".

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset 

# Parameters
WINDOW_SIZE = 100  # this is roughly 1 second

# ---------------------------
# Models
# ---------------------------
class SEMGCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)

        self.fc1 = nn.Linear(32 * (WINDOW_SIZE // 2), 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class SEMGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SEMGMLP(nn.Module):
    def __init__(self, n_classes, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, n_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ---------------------------
# Feature presets (modular)
# ---------------------------
FEATURE_PRESETS = {
    "baseline": {
        "add_wamp": False,
        "zc_threshold": 0.01,
        "ssc_threshold": 0.01,
        "wamp_threshold": 0.02,
    },
    "baseline_plus_wamp": {
        "add_wamp": True,
        "zc_threshold": 0.01,
        "ssc_threshold": 0.01,
        "wamp_threshold": 0.02,
    },
    # You can add more presets later
}


# ---------------------------
# Feature functions
# ---------------------------
def _mav(x):
    return np.mean(np.abs(x))

def _rms(x):
    return np.sqrt(np.mean(x ** 2))

def _wl(x):
    return np.sum(np.abs(np.diff(x)))

def _var(x):
    return np.var(x)

def _zc(x, threshold=0.01):
    return np.sum(
        ((x[:-1] * x[1:]) < 0) &
        (np.abs(x[:-1] - x[1:]) > threshold)
    )

def _ssc(x, threshold=0.01):
    return np.sum(
        ((x[1:-1] - x[:-2]) * (x[1:-1] - x[2:]) > 0) &
        (np.abs(x[1:-1] - x[:-2]) > threshold) &
        (np.abs(x[1:-1] - x[2:]) > threshold)
    )

def _wamp(x, threshold=0.02):
    return np.sum(np.abs(np.diff(x)) > threshold)


# ---------------------------
# Baseline extractor (your original)
# ---------------------------
def extract_features_baseline(window, zc_threshold=0.01, ssc_threshold=0.01):
    """
    Baseline classic sEMG features for each channel.
    window: np.ndarray shape (channels, window_size)
    Returns: np.ndarray shape (channels * 6,)
    Features: MAV, RMS, WL, VAR, ZC, SSC
    """
    feats = []
    for ch in window:
        feats.extend([
            _mav(ch),
            _rms(ch),
            _wl(ch),
            _var(ch),
            _zc(ch, threshold=zc_threshold),
            _ssc(ch, threshold=ssc_threshold),
        ])
    return np.array(feats, dtype=np.float32)


# ---------------------------
# Configurable extractor
# ---------------------------
def extract_features(window, preset="baseline", config=None):
    """
    Configurable feature extractor.
    Keeps backward compatibility: extract_features(window) still works.
    
    window: np.ndarray, shape (channels, window_size)
    preset: str, one of FEATURE_PRESETS keys
    config: optional dict overrides preset values
    """
    if config is None:
        if preset not in FEATURE_PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Available: {list(FEATURE_PRESETS.keys())}")
        config = FEATURE_PRESETS[preset]

    # Start with baseline (your original features)
    feats = extract_features_baseline(
        window,
        zc_threshold=config.get("zc_threshold", 0.01),
        ssc_threshold=config.get("ssc_threshold", 0.01),
    ).tolist()

    # Optional additions
    if config.get("add_wamp", False):
        wamp_threshold = config.get("wamp_threshold", 0.02)
        for ch in window:
            feats.append(_wamp(ch, threshold=wamp_threshold))

    return np.array(feats, dtype=np.float32)

How to use it in your training script
1) Import FEATURE_PRESETS

Change this line:

from semg_model import SEMGMLP, extract_features, WINDOW_SIZE, SEMGDataset

to:

from semg_model import SEMGMLP, extract_features, WINDOW_SIZE, SEMGDataset, FEATURE_PRESETS

2) Add a CLI argument for the preset

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="combined.csv")
parser.add_argument(
    "--feature_preset",
    default="baseline",
    choices=list(FEATURE_PRESETS.keys())
)
args = parser.parse_args()

3) Pass preset into create_samples

Update your create_samples:

def create_samples(df, window_size, step=STEP_SIZE, feature_preset="baseline"):
    window_size = int(window_size)
    step = int(step)
    X, y = [], []
    for i in range(0, len(df) - window_size + 1, step):
        chunk = df.iloc[i:i+window_size]
        label = chunk['Action'].iloc[0]
        if (chunk['Action'] != label).any():
            continue

        sample = chunk[['Ch0 Act', 'Ch1 Act', 'Ch2 Act', 'Ch3 Act']].values.T
        feats = extract_features(sample, preset=feature_preset)
        X.append(feats)
        y.append(label)

    return np.stack(X), np.array(y)

And call it like:

X, y = create_samples(df, WINDOW_SIZE, feature_preset=args.feature_preset)
print("Feature preset:", args.feature_preset)
print("Feature dimension:", X.shape[1])

How to use it in inference (important)

Training and inference must use the same preset.

In your detection script, also add --feature_preset and use:

feats = extract_features(window, preset=args.feature_preset)

And build input_dim dynamically (so you don’t hardcode 4 * 6 anymore):

dummy = np.zeros((4, int(WINDOW_SIZE)), dtype=np.float32)
input_dim = extract_features(dummy, preset=args.feature_preset).shape[0]

This avoids shape mismatch when you add WAMP.
Expected feature dimensions

With 4 channels:

    baseline = 4 × 6 = 24

    baseline_plus_wamp = 4 × 7 = 28

So if you use WAMP, the MLP input size becomes 28.
Should you recollect data?

Maybe, but do this first:

    Try baseline_plus_wamp

    Print windows per class (train/test)

    Check confusion matrix for flexion ↔ grasp confusion specifically

If flexion/grasp still struggle and counts are low → yes, recollect those classes (especially with variation: speed, intensity, wrist angle).

If you want, I can also give you a drop-in class-weighted loss snippet next (very helpful if flexion/grasp have fewer windows).

You said:
i haven't changed any of my code, tell me what to do from the start 

ChatGPT said:

Got you — here’s the cleanest way from the start, with minimal risk and easy rollback.
Goal

You want to:

    keep your current baseline features

    optionally add WAMP

    switch between them with a setting

    keep train + inference consistent

Step 1 — Replace semg_model.py with a modular version

Copy your current file, save a backup first (important):

    semg_model.py → semg_model_backup.py

Then replace semg_model.py with this:

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset 

# Parameters
WINDOW_SIZE = 100  # this is roughly 1 second

# ---------------------------
# Models
# ---------------------------
class SEMGCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)

        self.fc1 = nn.Linear(32 * (WINDOW_SIZE // 2), 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

class SEMGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SEMGMLP(nn.Module):
    def __init__(self, n_classes, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, n_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ---------------------------
# Feature presets
# ---------------------------
FEATURE_PRESETS = {
    "baseline": {
        "add_wamp": False,
        "zc_threshold": 0.01,
        "ssc_threshold": 0.01,
        "wamp_threshold": 0.02,
    },
    "baseline_plus_wamp": {
        "add_wamp": True,
        "zc_threshold": 0.01,
        "ssc_threshold": 0.01,
        "wamp_threshold": 0.02,
    },
}

# ---------------------------
# Feature functions
# ---------------------------
def _mav(x):
    return np.mean(np.abs(x))

def _rms(x):
    return np.sqrt(np.mean(x ** 2))

def _wl(x):
    return np.sum(np.abs(np.diff(x)))

def _var(x):
    return np.var(x)

def _zc(x, threshold=0.01):
    return np.sum(
        ((x[:-1] * x[1:]) < 0) &
        (np.abs(x[:-1] - x[1:]) > threshold)
    )

def _ssc(x, threshold=0.01):
    return np.sum(
        ((x[1:-1] - x[:-2]) * (x[1:-1] - x[2:]) > 0) &
        (np.abs(x[1:-1] - x[:-2]) > threshold) &
        (np.abs(x[1:-1] - x[2:]) > threshold)
    )

def _wamp(x, threshold=0.02):
    return np.sum(np.abs(np.diff(x)) > threshold)

# ---------------------------
# Baseline features (same as your current)
# ---------------------------
def extract_features_baseline(window, zc_threshold=0.01, ssc_threshold=0.01):
    feats = []
    for ch in window:
        feats.extend([
            _mav(ch),
            _rms(ch),
            _wl(ch),
            _var(ch),
            _zc(ch, threshold=zc_threshold),
            _ssc(ch, threshold=ssc_threshold),
        ])
    return np.array(feats, dtype=np.float32)

# ---------------------------
# Main feature extractor
# ---------------------------
def extract_features(window, preset="baseline", config=None):
    if config is None:
        if preset not in FEATURE_PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Available: {list(FEATURE_PRESETS.keys())}")
        config = FEATURE_PRESETS[preset]

    feats = extract_features_baseline(
        window,
        zc_threshold=config.get("zc_threshold", 0.01),
        ssc_threshold=config.get("ssc_threshold", 0.01),
    ).tolist()

    if config.get("add_wamp", False):
        wamp_threshold = config.get("wamp_threshold", 0.02)
        for ch in window:
            feats.append(_wamp(ch, threshold=wamp_threshold))

    return np.array(feats, dtype=np.float32)

Step 2 — Update your training script (small changes only)

You don’t need to rewrite everything. Just make these changes.
A) Update import

Change this:

from semg_model import SEMGMLP, extract_features, WINDOW_SIZE, SEMGDataset

to:

from semg_model import SEMGMLP, extract_features, WINDOW_SIZE, SEMGDataset, FEATURE_PRESETS

B) Add feature preset arg

Right after parser.add_argument("--data", default="combined.csv"), add:

parser.add_argument(
    "--feature_preset",
    default="baseline",
    choices=list(FEATURE_PRESETS.keys())
)

C) Make WINDOW_SIZE and STEP_SIZE int

Change:

STEP_SIZE = int(WINDOW_SIZE)

to:

WINDOW_SIZE = int(WINDOW_SIZE)
STEP_SIZE = int(WINDOW_SIZE)   # no overlap
# STEP_SIZE = int(WINDOW_SIZE // 4)  # use this later if you want 75% overlap

D) Update create_samples(...)

Replace your function with this version:

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

E) Call it with the preset

Change this:

X, y = create_samples(df, WINDOW_SIZE)

to:

X, y = create_samples(df, WINDOW_SIZE, feature_preset=args.feature_preset)
print("Feature preset:", args.feature_preset)
print("Feature dimension:", X.shape[1])

F) Print windows per class (ALL / TRAIN / TEST)

Add this after y_encoded = le.fit_transform(y):

all_counts = np.bincount(y_encoded, minlength=len(le.classes_))
print("\nWindows per class (ALL):")
for i, cls in enumerate(le.classes_):
    print(f"  {cls}: {all_counts[i]}")

Add this after train_test_split(...):

train_counts = np.bincount(y_train, minlength=len(le.classes_))
test_counts  = np.bincount(y_test, minlength=len(le.classes_))

print("\nWindows per class (TRAIN):")
for i, cls in enumerate(le.classes_):
    print(f"  {cls}: {train_counts[i]}")

print("\nWindows per class (TEST):")
for i, cls in enumerate(le.classes_):
    print(f"  {cls}: {test_counts[i]}")

Step 3 — Update your detection script (so feature dimensions match)

This is very important. If you train with baseline_plus_wamp, inference must use that too.
A) Update import

Change this:

from semg_model import extract_features, SEMGMLP, WINDOW_SIZE

to:

from semg_model import extract_features, SEMGMLP, WINDOW_SIZE, FEATURE_PRESETS

B) Add feature preset arg

In your parser, add:

parser.add_argument(
    "--feature_preset",
    default="baseline",
    choices=list(FEATURE_PRESETS.keys())
)

C) Parse args BEFORE creating model

Right now your script loads the model at the top before parsing args. Move the model-loading block below args = parser.parse_args().
Replace your current top model-loading section with this structure:

# parse args first
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["online", "offline"], default="online", help="Detection mode")
parser.add_argument("--csv", type=str, help="CSV file for offline mode")
parser.add_argument("--feature_preset", default="baseline", choices=list(FEATURE_PRESETS.keys()))
args = parser.parse_args()

WINDOW_SIZE = int(WINDOW_SIZE)

# Load label encoder
le = joblib.load("label_encoder_mlp.pkl")

# Build model with correct input dim based on feature preset
dummy_window = np.zeros((4, WINDOW_SIZE), dtype=np.float32)
input_dim = extract_features(dummy_window, preset=args.feature_preset).shape[0]
n_classes = len(le.classes_)

model = SEMGMLP(n_classes, input_dim)
model.load_state_dict(torch.load("semg_mlp.pth", map_location="cpu"))
model.eval()

# Initialize buffer AFTER WINDOW_SIZE is int
buffer = deque(maxlen=WINDOW_SIZE)

D) Use preset in prediction

In process_and_predict(data), change:

feats = extract_features(window)

to:

feats = extract_features(window, preset=args.feature_preset)

Step 4 — Train baseline first (sanity check)

Run your current baseline with the new modular setup:

python semg_train_mlp.py --data combined.csv --feature_preset baseline

You should see:

    feature dimension = 24

    windows per class printed

If this works, your refactor is safe.
Step 5 — Train with WAMP added

Delete old model files first (your script prevents overwrite):

    semg_mlp.pth

    label_encoder_mlp.pkl

Then run:

python semg_train_mlp.py --data combined.csv --feature_preset baseline_plus_wamp

You should see:

    feature dimension = 28

Step 6 — Evaluate offline with matching preset

If you trained with baseline_plus_wamp, evaluate with the same:

python semg_detect.py --mode offline --csv "your_test_files*.csv" --feature_preset baseline_plus_wamp

If you trained with baseline, use baseline.
Step 7 — Compare flexion/grasp specifically

Compare:

    per-class accuracy (flexion, grasp)

    confusion matrix row for flexion/grasp

If they improve → keep WAMP.
If not → next step is either:

    class-weighted loss

    more flexion/grasp data

    overlap windows (STEP_SIZE = WINDOW_SIZE // 4)

Quick rollback

If anything breaks:

    restore semg_model_backup.py → semg_model.py

    use --feature_preset baseline

If you want, I can also give you the exact updated training script and exact updated detection script in full so you can copy-paste without manually editing line by line.

You said:
import os
import numpy as np
import torch
from collections import deque
import socketio
import time
from semg_model import extract_features, SEMGMLP, WINDOW_SIZE, FEATURE_PRESETS
import joblib
import argparse
import pandas as pd
import sys
from sklearn.metrics import confusion_matrix
import datetime



def process_and_predict(data):
    try:
        raw = np.array(
            [
                data["ch0"]["a"],
                # data["ch0"]["e"],
                data["ch1"]["a"],
                # data["ch1"]["e"],
                data["ch2"]["a"],
                # data["ch2"]["e"],
                data["ch3"]["a"],
                # data["ch3"]["e"],
            ]
        )
        buffer.append(raw)
        if len(buffer) == WINDOW_SIZE:
            window = np.stack(buffer, axis=1)
            feats = extract_features(window)
            arr = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                out = model(arr)
                pred = out.argmax(dim=1).item()
                buffer.clear()
                return le.inverse_transform([pred])[0]
    except Exception as e:
        print(f"Inference error: {e}")
    return None


sio = socketio.Client()


@sio.event
def connect():
    print("Connected to Socket.IO server.")


@sio.on("adc_data")
def semg_data(data):
    prediction = process_and_predict(data)
    if prediction:
        print(f"Detection: {prediction}")


def offline_predict(csv_path):
    if "*" in csv_path:
        import glob
        csv_files = glob.glob(csv_path)
        if not csv_files:
            print("No CSV files found matching the pattern.")
            return
    else:
        csv_files = [csv_path]

    correct = 0

    # Initialize counters
    TP = {cls: 0 for cls in le.classes_}
    FP = {cls: 0 for cls in le.classes_}
    Total_per_class = {cls: 0 for cls in le.classes_}
    total_predictions = 0
    
    # Store all predictions for confusion matrix and CSV output
    all_actuals = []
    all_predictions = []
    prediction_records = []

    for csv_path in csv_files:
        buffer.clear()
        basename = os.path.basename(csv_path)
        ground_truth = None
        for cls in le.classes_:
            if cls in basename:
                ground_truth = cls
                break

        # print(f"Processing offline data from {basename}")
        df = pd.read_csv(csv_path)
        # Expecting columns: ch0_a, ch1_a, ch2_a, ch3_a
        for i in range(len(df)):
            data = {
                "ch0": {"a": df.iloc[i]["Ch0 Act"]},
                "ch1": {"a": df.iloc[i]["Ch1 Act"]},
                "ch2": {"a": df.iloc[i]["Ch2 Act"]},
                "ch3": {"a": df.iloc[i]["Ch3 Act"]},
            }
            pred = process_and_predict(data)
            if pred:
                total_predictions += 1
                # print(f"Offline Detection at row {i}: {pred}")
                
                # Record prediction
                all_actuals.append(ground_truth)
                all_predictions.append(pred)
                prediction_records.append({
                    'file': basename,
                    'actual': ground_truth,
                    'predicted': pred
                })
                
                if pred == ground_truth:
                    correct += 1
                    TP[ground_truth] += 1
                else:
                    FP[pred] += 1
                
                Total_per_class[ground_truth] += 1
        buffer.clear()

    # Accuracy
    accuracy = correct / total_predictions if total_predictions > 0 else 0.0

    # Per-class metrics
    print("\nMetrics per class:")
    for cls in le.classes_:
        class_accuracy = TP[cls]  / Total_per_class[cls] if Total_per_class[cls] > 0 else 0.0

        print(f"Class: {cls}")
        print(f"  Accuracy:  {class_accuracy:.4f}")

    print("\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}\n")
    
    # Save predictions to CSV
    if prediction_records:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"predictions_{timestamp}.csv"
        results_df = pd.DataFrame(prediction_records)
        results_df.to_csv(results_filename, index=False)
        print(f"Predictions saved to: {results_filename}")
    
    # Display confusion matrix
    if all_actuals and all_predictions:
        cm = confusion_matrix(all_actuals, all_predictions, labels=le.classes_)
        cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
        
        print("\nConfusion Matrix:")
        print("(Rows: Actual, Columns: Predicted)")
        print(cm_df)
        
        # Save confusion matrix to CSV
        cm_filename = f"confusion_matrix_{timestamp}.csv"
        cm_df.to_csv(cm_filename)
        print(f"\nConfusion matrix saved to: {cm_filename}")
    


if __name__ == "__main__":

    print(" ".join(sys.argv))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["online", "offline"],
        default="online",
        help="Detection mode",
    )
    parser.add_argument("--csv", type=str, help="CSV file for offline mode")
    parser.add_argument(
        "--feature_preset",
        default="baseline",
        choices=list(FEATURE_PRESETS.keys())
    )
    args = parser.parse_args()

    # Ensure WINDOW_SIZE is int
    WINDOW_SIZE_INT = int(WINDOW_SIZE)

    # Load label encoder
    le = joblib.load("label_encoder_mlp.pkl")

    # Build model with correct input dim based on feature preset
    dummy_window = np.zeros((4, WINDOW_SIZE_INT), dtype=np.float32)
    input_dim = extract_features(dummy_window, preset=args.feature_preset).shape[0]
    n_classes = len(le.classes_)
    model = SEMGMLP(n_classes, input_dim)
    model.load_state_dict(torch.load("semg_mlp.pth", map_location="cpu"))
    model.eval()

    # Initialize buffer AFTER WINDOW_SIZE is int
    buffer = deque(maxlen=WINDOW_SIZE_INT)

    if args.mode == "online":
        sio.connect("http://localhost:3000", transports=["websocket"], namespaces=["/"])
        try:
            print("waiting...")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            sio.disconnect()
    elif args.mode == "offline":
        if not args.csv:
            print("Please provide a CSV file with --csv for offline mode.")
        else:
            offline_predict(args.csv)

is this step correct for c)?

ChatGPT said:

Yes — this is mostly correct, and you’ve done the important part for (c): you moved arg parsing + model creation into if __name__ == "__main__": and made the input dim depend on the selected feature preset. ✅

There’s just one key fix left:
Fix: process_and_predict() must use the selected preset

Right now it still does:

feats = extract_features(window)

That ignores --feature_preset and always uses the default (baseline).
Change it to:

feats = extract_features(window, preset=args.feature_preset)

Small but important window-size consistency fix

Inside process_and_predict(), you compare against WINDOW_SIZE (from import), but your buffer uses WINDOW_SIZE_INT.

Safer to use the same one everywhere.
Change:

if len(buffer) == WINDOW_SIZE:

To:

if len(buffer) == WINDOW_SIZE_INT:

(Or simpler: overwrite WINDOW_SIZE = int(WINDOW_SIZE) once globally, but your current approach is fine.)
Why this still works even though args/model/buffer are defined later

It’s okay because:

    process_and_predict() is defined early

    but only called after args, model, buffer, le are created in __main__

So Python resolves those names at runtime. 👍
Recommended final patch (only 2 edits)

Inside process_and_predict():

if len(buffer) == WINDOW_SIZE_INT:
    window = np.stack(buffer, axis=1)
    feats = extract_features(window, preset=args.feature_preset)

Optional (nice cleanup)

To avoid any confusion, add these globals at the top (not required, just clearer):

model = None
le = None
buffer = None
args = None
WINDOW_SIZE_INT = None

Quick test you should run

After loading model in __main__, print:

print("Feature preset:", args.feature_preset)
print("Model input_dim:", input_dim)

Expected:

    baseline → 24

    baseline_plus_wamp → 28

So yes — your step (c) is correct, just apply those 2 fixes in process_and_predict() and you’re good.

You said:
import subprocess
import sys
import os
import shutil
import re

NUM_ITERATIONS = 20
DATA_PATH = r"..\sutd_bmi_safety_data\combined.csv"
UNSEEN_CSV = r"..\sutd_bmi_safety_data\unseen\*.csv"

MODEL_FILE = "semg_mlp.pth"
ENCODER_FILE = "label_encoder_mlp.pkl"
BEST_MODEL_FILE = "semg_mlp_best.pth"
BEST_ENCODER_FILE = "label_encoder_mlp_best.pkl"

best_unseen_accuracy = 0.0
best_val_accuracy = 0.0
best_iteration = -1

# Store results for all iterations
all_results = []

for i in range(NUM_ITERATIONS):
    print(f"\n{'='*60}")
    print(f"Iteration {i+1}/{NUM_ITERATIONS}")
    print(f"{'='*60}")

    # Remove existing model so training script doesn't skip
    for f in [MODEL_FILE, ENCODER_FILE]:
        if os.path.exists(f):
            os.remove(f)

    # Train
    train_result = subprocess.run(
        [sys.executable, "semg_train_mlp.py", "--data", DATA_PATH],
        capture_output=True, text=True
    )
    print(train_result.stdout[-300:] if len(train_result.stdout) > 300 else train_result.stdout)
    if train_result.returncode != 0:
        print(f"Training failed:\n{train_result.stderr}")
        continue

    # Parse validation accuracy from training output (e.g. "Best model saved with accuracy: 95.00%")
    val_match = re.search(r"Best model saved with accuracy:\s+([\d.]+)%", train_result.stdout)
    val_acc = float(val_match.group(1)) / 100.0 if val_match else 0.0

    # Infer on unseen data
    infer_result = subprocess.run(
        [sys.executable, "semg_infer_mlp.py", "--mode", "offline", "--csv", UNSEEN_CSV],
        capture_output=True, text=True
    )
    print(infer_result.stdout[-500:] if len(infer_result.stdout) > 500 else infer_result.stdout)
    if infer_result.returncode != 0:
        print(f"Inference failed:\n{infer_result.stderr}")
        continue

    # Parse overall accuracy from inference output
    all_matches = re.findall(r"Accuracy:\s+([\d.]+)", infer_result.stdout)
    if not all_matches:
        print("Could not parse accuracy from inference output.")
        continue
    unseen_acc = float(all_matches[-1])

    # Store results
    all_results.append({
        "iteration": i + 1,
        "val_acc": val_acc,
        "unseen_acc": unseen_acc,
    })

    print(f"\n>>> Iteration {i+1}: Val Acc: {val_acc:.4f} | Unseen Acc: {unseen_acc:.4f} (best unseen so far: {best_unseen_accuracy:.4f})")

    if unseen_acc > best_unseen_accuracy:
        best_unseen_accuracy = unseen_acc
        best_val_accuracy = val_acc
        best_iteration = i + 1
        shutil.copy(MODEL_FILE, BEST_MODEL_FILE)
        shutil.copy(ENCODER_FILE, BEST_ENCODER_FILE)
        print(f">>> NEW BEST! Saved as {BEST_MODEL_FILE}")

# ============================================================
# Final Summary
# ============================================================
print(f"\n{'='*60}")
print(f"  SEARCH COMPLETE - SUMMARY")
print(f"{'='*60}")
print(f"\nAll iterations:")
print(f"{'Iter':<6} {'Val Acc':<12} {'Unseen Acc':<12} {'Best?'}")
print(f"{'-'*42}")
for r in all_results:
    marker = " <<<" if r["iteration"] == best_iteration else ""
    print(f"{r['iteration']:<6} {r['val_acc']:<12.4f} {r['unseen_acc']:<12.4f}{marker}")

print(f"\n{'='*60}")
print(f"  BEST MODEL (Iteration {best_iteration})")
print(f"{'='*60}")
print(f"  Validation Accuracy (train split): {best_val_accuracy:.2%}")
print(f"  Unseen Data Accuracy:              {best_unseen_accuracy:.2%}")
print(f"  Model saved as:  {BEST_MODEL_FILE}")
print(f"  Encoder saved as: {BEST_ENCODER_FILE}")
print(f"{'='*60}")

# Run final inference with the best model to show detailed results
print(f"\n{'='*60}")
print(f"  DETAILED RESULTS FROM BEST MODEL")
print(f"{'='*60}")
shutil.copy(BEST_MODEL_FILE, MODEL_FILE)
shutil.copy(BEST_ENCODER_FILE, ENCODER_FILE)
final_result = subprocess.run(
    [sys.executable, "semg_infer_mlp.py", "--mode", "offline", "--csv", UNSEEN_CSV],
    capture_output=True, text=True
)
print(final_result.stdout)

how do i fix this script, does it even make sense to run this ? for itteravtive testing

ChatGPT said:

Yes, it makes sense technically (it’s doing repeated training and picking the best run), but there’s a big ML caveat:
The main issue

You are selecting the “best” model using the unseen dataset (UNSEEN_CSV).

That means the unseen set is no longer truly unseen — it becomes a tuning set.
So your final unseen accuracy will look better than reality.
Better practice

    Use train split + validation split for iterative testing / model selection

    Use the unseen set only once at the end for final evaluation

What your script is doing now (and what’s okay)

It:

    trains model multiple times

    evaluates on unseen data each time

    keeps best unseen model

This is okay for debugging/engineering (e.g., “does WAMP help?”), but not okay for reporting final performance.
What to fix in your script (practical)
1) Pass --feature_preset through train and infer

You added modular features, but this search script doesn’t pass the preset yet.
2) Add timeouts (so it doesn’t hang)

Earlier you had subprocess hangs. Add timeout=....
3) Print full stderr when parsing fails

Good for debugging.
4) Parse accuracies more robustly

Your inference output contains:

    per-class Accuracy: ...

    overall Accuracy: ...

You currently take the last one, which is okay if overall is always printed last, but better to parse the “Overall Metrics” section more explicitly.
5) (Important) If you want true iterative testing:

Use unseen only for final run, and choose best model by validation accuracy.
Recommended version (safe + flexible)

This version:

    supports --feature_preset

    supports --select_on (val or unseen)

    adds timeout

    avoids silent failures

import subprocess
import sys
import os
import shutil
import re
import argparse

NUM_ITERATIONS = 20
DATA_PATH = r"..\sutd_bmi_safety_data\combined.csv"
UNSEEN_CSV = r"..\sutd_bmi_safety_data\unseen\*.csv"

MODEL_FILE = "semg_mlp.pth"
ENCODER_FILE = "label_encoder_mlp.pkl"
BEST_MODEL_FILE = "semg_mlp_best.pth"
BEST_ENCODER_FILE = "label_encoder_mlp_best.pkl"

def parse_val_acc(train_stdout: str) -> float:
    # Looks for: "Best model saved with accuracy: 95.00%"
    m = re.search(r"Best model saved with accuracy:\s*([\d.]+)%", train_stdout)
    return float(m.group(1)) / 100.0 if m else None

def parse_unseen_acc(infer_stdout: str) -> float:
    # Prefer parsing the "Overall Metrics" block
    m = re.search(r"Overall Metrics:\s*[\r\n]+.*?Accuracy:\s*([\d.]+)", infer_stdout, re.DOTALL)
    if m:
        return float(m.group(1))
    # Fallback: last "Accuracy: X"
    all_matches = re.findall(r"Accuracy:\s*([\d.]+)", infer_stdout)
    return float(all_matches[-1]) if all_matches else None

def tail(s, n=700):
    return s[-n:] if s and len(s) > n else (s or "")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=NUM_ITERATIONS)
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--unseen_csv", default=UNSEEN_CSV)
    parser.add_argument("--feature_preset", default="baseline")
    parser.add_argument(
        "--select_on",
        choices=["val", "unseen"],
        default="val",
        help="Use 'val' for proper model selection. 'unseen' is only for quick debugging."
    )
    parser.add_argument("--timeout_sec", type=int, default=600)
    args = parser.parse_args()

    best_score = -1.0
    best_unseen_accuracy = 0.0
    best_val_accuracy = 0.0
    best_iteration = -1
    all_results = []

    for i in range(args.iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {i+1}/{args.iterations}")
        print(f"{'='*60}")

        # Remove existing model so training script doesn't skip
        for f in [MODEL_FILE, ENCODER_FILE]:
            if os.path.exists(f):
                os.remove(f)

        # ---------------- Train ----------------
        try:
            train_cmd = [
                sys.executable, "semg_train_mlp.py",
                "--data", args.data,
                "--feature_preset", args.feature_preset,
            ]
            train_result = subprocess.run(
                train_cmd,
                capture_output=True,
                text=True,
                timeout=args.timeout_sec
            )
        except subprocess.TimeoutExpired:
            print(f"Training timed out after {args.timeout_sec}s")
            continue

        print(tail(train_result.stdout, 800))
        if train_result.returncode != 0:
            print("Training failed:")
            print(train_result.stderr)
            continue

        val_acc = parse_val_acc(train_result.stdout)
        if val_acc is None:
            print("Could not parse validation accuracy from training output.")
            continue

        # ---------------- Infer on unseen ----------------
        try:
            infer_cmd = [
                sys.executable, "semg_infer_mlp.py",
                "--mode", "offline",
                "--csv", args.unseen_csv,
                "--feature_preset", args.feature_preset,
            ]
            infer_result = subprocess.run(
                infer_cmd,
                capture_output=True,
                text=True,
                timeout=args.timeout_sec
            )
        except subprocess.TimeoutExpired:
            print(f"Inference timed out after {args.timeout_sec}s")
            continue

        print(tail(infer_result.stdout, 1000))
        if infer_result.returncode != 0:
            print("Inference failed:")
            print(infer_result.stderr)
            continue

        unseen_acc = parse_unseen_acc(infer_result.stdout)
        if unseen_acc is None:
            print("Could not parse unseen accuracy from inference output.")
            continue

        all_results.append({
            "iteration": i + 1,
            "val_acc": val_acc,
            "unseen_acc": unseen_acc,
        })

        # Choose selection metric
        score = val_acc if args.select_on == "val" else unseen_acc

        print(
            f"\n>>> Iteration {i+1}: "
            f"Val Acc: {val_acc:.4f} | Unseen Acc: {unseen_acc:.4f} "
            f"| Selection score ({args.select_on}): {score:.4f}"
        )

        if score > best_score:
            best_score = score
            best_val_accuracy = val_acc
            best_unseen_accuracy = unseen_acc
            best_iteration = i + 1

            shutil.copy(MODEL_FILE, BEST_MODEL_FILE)
            shutil.copy(ENCODER_FILE, BEST_ENCODER_FILE)
            print(f">>> NEW BEST! Saved as {BEST_MODEL_FILE}")

    # ---------------- Summary ----------------
    print(f"\n{'='*60}")
    print("SEARCH COMPLETE - SUMMARY")
    print(f"{'='*60}")

    if not all_results:
        print("No successful iterations.")
        return

    print(f"\nFeature preset: {args.feature_preset}")
    print(f"Selection metric: {args.select_on}")
    print(f"{'Iter':<6} {'Val Acc':<12} {'Unseen Acc':<12} {'Best?'}")
    print("-" * 46)
    for r in all_results:
        marker = "<<<" if r["iteration"] == best_iteration else ""
        print(f"{r['iteration']:<6} {r['val_acc']:<12.4f} {r['unseen_acc']:<12.4f} {marker}")

    print(f"\n{'='*60}")
    print(f"BEST MODEL (Iteration {best_iteration})")
    print(f"{'='*60}")
    print(f"Validation Accuracy: {best_val_accuracy:.2%}")
    print(f"Unseen Accuracy:     {best_unseen_accuracy:.2%}")
    print(f"Model saved as:      {BEST_MODEL_FILE}")
    print(f"Encoder saved as:    {BEST_ENCODER_FILE}")
    print(f"{'='*60}")

    # Final detailed run with best model
    print(f"\n{'='*60}")
    print("DETAILED RESULTS FROM BEST MODEL")
    print(f"{'='*60}")
    shutil.copy(BEST_MODEL_FILE, MODEL_FILE)
    shutil.copy(BEST_ENCODER_FILE, ENCODER_FILE)

    final_cmd = [
        sys.executable, "semg_infer_mlp.py",
        "--mode", "offline",
        "--csv", args.unseen_csv,
        "--feature_preset", args.feature_preset,
    ]
    final_result = subprocess.run(final_cmd, capture_output=True, text=True)
    print(final_result.stdout)
    if final_result.returncode != 0:
        print(final_result.stderr)

if __name__ == "__main__":
    main()

Does iterative testing even help here?

Yes, but only if there is randomness. Right now your training script has:

    train_test_split(... random_state=42) → fixed split ✅

    but no fixed torch seed → model init / dataloader shuffle can vary ✅

So repeated runs can produce slightly different results.
Better than “20 random retries”

If your goal is improving flexion/grasp, these are more useful than repeated reruns:

    Feature set changes (baseline vs baseline_plus_wamp)

    Class-weighted loss

    More flexion/grasp windows

    Cross-validation (better than lucky single split)

Best practice workflow (recommended)

Use your search script like this:
For model selection (proper)

python semg_search_best.py --feature_preset baseline_plus_wamp --select_on val

Final report

Then run unseen once on the chosen model.
One more thing you should consider

Your current training script exits if model file exists:

if os.path.exists('semg_mlp.pth'):
    ...
    exit(0)

Your search script deletes it, so that’s fine. But if a run fails before saving, next steps can break. The updated script handles that better by checking return codes.

If you want, I can also help you make this search script compare multiple presets automatically (baseline vs baseline_plus_wamp) and print a side-by-side summary.

You said:
Use unseen only for final run, and choose best model by validation accuracy.
what do you mean

ChatGPT said:

It means: don’t use your unseen CSVs to decide which model is “best.”
Use the validation score from training to pick the best run, then test on unseen once at the end.
Why

If you keep checking unseen accuracy during tuning, you slowly “optimize for the unseen set” (even indirectly). Then it’s no longer a true final test.

Think of it like:

    Train set = practice

    Validation set = mock exam (you can use this to improve)

    Unseen/test set = real exam (you should only check once)

In your script

Right now you do:

    train

    run on unseen

    keep the run with highest unseen accuracy ❌

Better:

    train

    read validation accuracy from training output

    keep the run with highest validation accuracy ✅

    after all iterations, run the single best saved model on unseen once ✅

What to change (conceptually)

In your loop:

    still run training every iteration

    parse val_acc

    choose best using val_acc (not unseen_acc)

You can even skip unseen inference inside the loop entirely (faster + cleaner).
Simple example

If 3 runs give:

    Run 1: val 0.90, unseen 0.82

    Run 2: val 0.88, unseen 0.85

    Run 3: val 0.91, unseen 0.80

You should pick Run 3 (highest validation), then evaluate it on unseen once.
Even if Run 2 had a better unseen score, choosing it because of unseen means you’re tuning on the test set.
When is it okay to use unseen in the loop?

Only for debugging/experimentation, not for reporting final performance.

If you want, I can show you the exact lines in your current script to change so it selects by validation and only does unseen at the end.

You said:
okay do that change

ChatGPT said:

Perfect — here’s the clean change.
What you’re changing

    Inside the loop: train only, parse validation accuracy, pick best by val_acc

    After the loop: run unseen inference once using the best saved model

This is the proper setup.
Updated semg_search_best.py (validation-based selection, unseen only at end)

Replace your current script with this:

import subprocess
import sys
import os
import shutil
import re
import argparse

DEFAULT_NUM_ITERATIONS = 20
DEFAULT_DATA_PATH = r"..\sutd_bmi_safety_data\combined.csv"
DEFAULT_UNSEEN_CSV = r"..\sutd_bmi_safety_data\unseen\*.csv"

MODEL_FILE = "semg_mlp.pth"
ENCODER_FILE = "label_encoder_mlp.pkl"
BEST_MODEL_FILE = "semg_mlp_best.pth"
BEST_ENCODER_FILE = "label_encoder_mlp_best.pkl"


def parse_val_acc(train_stdout: str):
    # Parses: "Best model saved with accuracy: 95.00%"
    m = re.search(r"Best model saved with accuracy:\s*([\d.]+)%", train_stdout)
    return float(m.group(1)) / 100.0 if m else None


def parse_unseen_acc(infer_stdout: str):
    # Prefer "Overall Metrics" block if present
    m = re.search(r"Overall Metrics:\s*[\r\n]+.*?Accuracy:\s*([\d.]+)", infer_stdout, re.DOTALL)
    if m:
        return float(m.group(1))
    # Fallback: last occurrence of "Accuracy:"
    all_matches = re.findall(r"Accuracy:\s*([\d.]+)", infer_stdout)
    return float(all_matches[-1]) if all_matches else None


def tail(s: str, n: int = 800):
    if not s:
        return ""
    return s[-n:] if len(s) > n else s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=DEFAULT_NUM_ITERATIONS)
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--unseen_csv", default=DEFAULT_UNSEEN_CSV)
    parser.add_argument("--feature_preset", default="baseline")
    parser.add_argument("--timeout_sec", type=int, default=600)
    args = parser.parse_args()

    best_val_accuracy = -1.0
    best_iteration = -1
    all_results = []

    # ------------------------------
    # Iterative training (select by validation ONLY)
    # ------------------------------
    for i in range(args.iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {i+1}/{args.iterations}")
        print(f"{'='*60}")

        # Remove current model artifacts so training script doesn't exit early
        for f in [MODEL_FILE, ENCODER_FILE]:
            if os.path.exists(f):
                os.remove(f)

        # Run training
        train_cmd = [
            sys.executable, "semg_train_mlp.py",
            "--data", args.data,
            "--feature_preset", args.feature_preset,
        ]

        try:
            train_result = subprocess.run(
                train_cmd,
                capture_output=True,
                text=True,
                timeout=args.timeout_sec
            )
        except subprocess.TimeoutExpired:
            print(f"Training timed out after {args.timeout_sec}s")
            continue

        print(tail(train_result.stdout, 1000))

        if train_result.returncode != 0:
            print("Training failed:")
            print(train_result.stderr)
            continue

        # Parse validation accuracy from training output
        val_acc = parse_val_acc(train_result.stdout)
        if val_acc is None:
            print("Could not parse validation accuracy from training output.")
            continue

        all_results.append({
            "iteration": i + 1,
            "val_acc": val_acc,
        })

        print(f">>> Iteration {i+1}: Val Acc = {val_acc:.4f} (best so far: {best_val_accuracy:.4f})")

        # Select best model by validation accuracy only
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_iteration = i + 1

            if os.path.exists(MODEL_FILE) and os.path.exists(ENCODER_FILE):
                shutil.copy(MODEL_FILE, BEST_MODEL_FILE)
                shutil.copy(ENCODER_FILE, BEST_ENCODER_FILE)
                print(f">>> NEW BEST by validation! Saved as {BEST_MODEL_FILE}")
            else:
                print("Warning: trained model files not found to copy.")

    # ------------------------------
    # Summary of training runs
    # ------------------------------
    print(f"\n{'='*60}")
    print("SEARCH COMPLETE - TRAINING SUMMARY")
    print(f"{'='*60}")

    if not all_results:
        print("No successful training iterations.")
        return

    print(f"\nFeature preset: {args.feature_preset}")
    print(f"{'Iter':<6} {'Val Acc':<12} {'Best?'}")
    print("-" * 30)
    for r in all_results:
        marker = "<<<" if r["iteration"] == best_iteration else ""
        print(f"{r['iteration']:<6} {r['val_acc']:<12.4f} {marker}")

    print(f"\nBest iteration (by validation): {best_iteration}")
    print(f"Best validation accuracy:       {best_val_accuracy:.2%}")
    print(f"Saved model:                   {BEST_MODEL_FILE}")
    print(f"Saved encoder:                 {BEST_ENCODER_FILE}")

    # ------------------------------
    # Final unseen evaluation (run ONCE)
    # ------------------------------
    if not (os.path.exists(BEST_MODEL_FILE) and os.path.exists(BEST_ENCODER_FILE)):
        print("\nBest model artifacts not found. Skipping final unseen evaluation.")
        return

    print(f"\n{'='*60}")
    print("FINAL EVALUATION ON UNSEEN DATA (RUN ONCE)")
    print(f"{'='*60}")

    # Copy best artifacts into the names expected by inference script
    shutil.copy(BEST_MODEL_FILE, MODEL_FILE)
    shutil.copy(BEST_ENCODER_FILE, ENCODER_FILE)

    infer_cmd = [
        sys.executable, "semg_infer_mlp.py",
        "--mode", "offline",
        "--csv", args.unseen_csv,
        "--feature_preset", args.feature_preset,
    ]

    try:
        infer_result = subprocess.run(
            infer_cmd,
            capture_output=True,
            text=True,
            timeout=args.timeout_sec
        )
    except subprocess.TimeoutExpired:
        print(f"Final unseen inference timed out after {args.timeout_sec}s")
        return

    print(infer_result.stdout)
    if infer_result.returncode != 0:
        print("Final inference failed:")
        print(infer_result.stderr)
        return

    unseen_acc = parse_unseen_acc(infer_result.stdout)
    if unseen_acc is not None:
        print(f"\nFinal unseen accuracy: {unseen_acc:.4f}")
    else:
        print("\nCould not parse final unseen accuracy (but detailed output printed above).")


if __name__ == "__main__":
    main()

How to run it
Baseline

python semg_search_best.py --feature_preset baseline

Baseline + WAMP

python semg_search_best.py --feature_preset baseline_plus_wamp

Then compare the final unseen accuracy from each run.
Why this is better

Now your script is doing proper model selection:

    validation set decides which run is best

    unseen set is used only once for final testing

That gives you a much more trustworthy unseen score.

If you want, next I can help you upgrade this script to compare both presets in one run and print a final side-by-side summary (baseline vs baseline_plus_wamp).

You said:
Best iteration (by validation): 10
Best validation accuracy:       94.87%
Saved model:                   semg_mlp_best.pth
Saved encoder:                 label_encoder_mlp_best.pkl

============================================================
FINAL EVALUATION ON UNSEEN DATA (RUN ONCE)
============================================================
semg_infer_mlp.py --mode offline --csv ..\sutd_bmi_safety_data\unseen\*.csv --feature_preset baseline
Feature preset: baseline
Model input_dim: 24

Metrics per class:
Class: extension
  Accuracy:  1.0000
Class: flexion
  Accuracy:  0.6667
Class: grasp
  Accuracy:  0.2222
Class: left
  Accuracy:  0.8889
Class: nothing
  Accuracy:  1.0000
Class: open
  Accuracy:  1.0000
Class: pronation
  Accuracy:  0.5556
Class: right
  Accuracy:  0.4444
Class: supination
  Accuracy:  0.8889

Overall Metrics:
  Accuracy:  0.7407
why so low

ChatGPT said:

That result actually makes sense — your validation is high (94.87%) but unseen is 74.07% because the unseen set is probably a different distribution from your train/val split.

In short: your model learned your training data well, but some gestures (especially grasp, flexion, right, pronation) don’t generalize as well.
Why this happens
1) Train/val split is easier than unseen data

Your validation set is created from the same combined.csv using a random split, so train and val likely contain:

    similar signal quality

    same session/device placement

    same user style

    similar timing/intensity

So val accuracy can look great.

Your unseen files are more realistic:

    different session

    slightly different electrode placement

    different muscle effort

    different gesture speed

    more noise / transitions

That gap is very common in EMG.
2) Some classes are naturally confusable

Your weak classes are:

    grasp: 0.2222

    flexion: 0.6667

    right: 0.4444

    pronation: 0.5556

These often have overlapping muscle activation patterns, especially if:

    the user performed them similarly

    labels include transition periods

    windows are too coarse (1s non-overlapping)

3) Windowing mismatch / too few predictions per class

Your per-class accuracies look like fractions of 9 (0.2222 = 2/9, 0.6667 = 6/9), so each class may only have ~9 windows in unseen.

That means:

    just a few wrong predictions tank the class accuracy

    one noisy file can hurt a lot

So the number looks harsh because the unseen set is small.
4) You’re using non-overlapping windows

In training you used:

STEP_SIZE = int(WINDOW_SIZE)

So no overlap.

That gives fewer windows and less variation. For EMG, overlap (e.g. 75%) usually helps a lot.
What to do next (in order)
A) Check the confusion matrix first

You already print it. Look specifically at:

    grasp → what is it being predicted as? (open? flexion?)

    right ↔ left

    pronation ↔ supination

That tells you what feature is missing.
B) Try baseline_plus_wamp

Since your baseline already has WL/ZC/SSC, WAMP is the next simplest improvement.

Run:

python semg_search_best.py --feature_preset baseline_plus_wamp

If grasp/flexion improve, keep it.
C) Use overlap windows in training (big win)

Change training step size from no overlap to 75% overlap:

STEP_SIZE = int(WINDOW_SIZE // 4)

This gives:

    more training samples

    better robustness

    more varied windows from each gesture

Keep inference as non-overlap if you want — that’s okay.
D) Add class-weighted loss (helps weak classes)

If grasp/flexion have fewer windows, weighted loss helps the model care more about them.

In your training script, after y_train is available:

class_counts = np.bincount(y_train, minlength=len(le.classes_))
class_weights = class_counts.sum() / (len(class_counts) * class_counts)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

This often helps minority / harder classes.
E) Collect more data for the weak classes (but targeted)

If you recollect, don’t just collect “more everything.”

Focus on:

    grasp

    flexion

    right

    pronation

And collect with variation:

    different force levels (light/medium/strong)

    slightly different arm/wrist angles

    multiple sessions (different days)

    clean “hold” periods (less transition noise)

This matters more than just volume.
F) Normalize per window or per channel (optional, helpful)

Sometimes EMG amplitude varies a lot across sessions.

Try normalizing each channel in each window before feature extraction:

ch = (ch - np.mean(ch)) / (np.std(ch) + 1e-8)

This can reduce session-to-session amplitude drift.

(If you do this, do it in both train and inference.)
G) Use grouped split later (better validation)

Your current random split can be too optimistic.

A better evaluation is:

    split by file/session (not random windows)

    so validation resembles “unseen” more closely

That will make val lower, but more honest.
What I’d do next (practical)

    Run baseline_plus_wamp

    Enable 75% overlap in training

    Add class-weighted loss

    Compare confusion matrices for grasp/flexion

That combo usually gives a meaningful bump.

If you want, I can give you the exact code patches for:

    overlap training windows

    weighted loss

    optional per-window normalization
    all in your current scripts (copy-paste ready).


You said:
Best iteration (by validation): 7
Best validation accuracy:       100.00%
Saved model:                   semg_mlp_best.pth
Saved encoder:                 label_encoder_mlp_best.pkl

============================================================
FINAL EVALUATION ON UNSEEN DATA (RUN ONCE)
============================================================
semg_infer_mlp.py --mode offline --csv ..\sutd_bmi_safety_data\unseen\*.csv --feature_preset baseline_plus_wamp
Feature preset: baseline_plus_wamp
Model input_dim: 28

Metrics per class:
Class: extension
  Accuracy:  1.0000
Class: flexion
  Accuracy:  0.7778
Class: grasp
  Accuracy:  0.4444
Class: left
  Accuracy:  0.6667
Class: nothing
  Accuracy:  1.0000
Class: open
  Accuracy:  1.0000
Class: pronation
  Accuracy:  0.8889
Class: right
  Accuracy:  0.4444
Class: supination
  Accuracy:  1.0000

Overall Metrics:
  Accuracy:  0.8025

Predictions saved to: predictions_20260225_001201.csv

Confusion Matrix:

ChatGPT said:

Nice — that’s a real improvement.

You went from 0.7407 → 0.8025 on unseen, and the biggest gains are exactly in the hard classes:

    grasp: 0.2222 → 0.4444

    flexion: 0.6667 → 0.7778

    pronation: 0.5556 → 0.8889

So adding WAMP clearly helped. ✅
But why is validation now 100%?

That’s a red flag for overfitting / easy split, not necessarily a bug.

Your validation is from random windows in the same combined.csv, so it’s likely too similar to training. With overlap or repeated patterns, the model can basically memorize signal style.
What it means

    The model is probably strong enough for your current data

    The unseen score (80.25%) is the more realistic number

    Your next gains will come from data split strategy + data quality + harder classes

What to do next (best next steps)
1) Use overlap windows in training

This is still worth doing if you haven’t already:

STEP_SIZE = int(WINDOW_SIZE // 4)   # 75% overlap

This usually improves robustness and gives more windows for hard classes.
2) Add class-weighted loss

Your weak classes are still:

    grasp

    right

    left (dropped here)

Weighted loss can push the model to care more about them.

Use:

class_counts = np.bincount(y_train, minlength=len(le.classes_))
class_weights = class_counts.sum() / (len(class_counts) * class_counts)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

3) Check the confusion matrix (super important)

You stopped at “Confusion Matrix:” — the actual matrix will tell you what grasp is being confused with.

Look for:

    grasp → open?

    right ↔ left?

    pronation ↔ supination?

That tells us what feature to add next.
4) Re-collect only the weak classes (targeted)

Now that your model is decent, extra data will help a lot — but focus on:

    grasp

    right

    left (it dropped)

    maybe flexion for consistency

Collect:

    multiple sessions

    different effort strengths

    slight wrist orientation changes

    clean holds (less transition)

5) Improve validation split (more honest)

If possible, split by recording file/session, not random windows.
Random window split can leak style/patterns across train and val.

That’s why you’re seeing 100% val but ~80% unseen.
Your current result is actually good

For multi-class sEMG with unseen session-style data, 80% unseen with a simple MLP + handcrafted features is solid.

And the fact that WAMP improved exactly the weak classes means you’re moving in the right direction.
Recommended next experiment (highest ROI)

Try this combo:

    baseline_plus_wamp

    75% overlap

    class-weighted loss

That’s probably your best next bump without changing the model architecture.

If you want, I can give you the exact copy-paste patches for:

    overlap training windows

    weighted loss

    per-class precision/recall/F1 printout (better than just accuracy)


You said:
does the overlap only need to be in the training.py?

ChatGPT said:

Yes — for now, overlap only in training.py is totally fine. ✅

That’s the usual setup:

    Training: use overlap (more samples, better learning)

    Inference: can stay non-overlap (simpler, fewer repeated predictions)

Why this works

Overlap in training helps the model see more varied windows from the same gesture.
It improves robustness without changing how deployment works.

Your inference pipeline still uses the same window size and same feature extraction, so input shape stays consistent.
When to also use overlap in inference

Only if you want:

    more frequent predictions (e.g., every 250 ms instead of every 1 s)

    smoother output (majority vote over recent predictions)

Right now your inference does:

buffer.clear()

That means non-overlap.

If later you want overlap inference, change that to slide by step size instead of clearing:

for _ in range(WINDOW_SIZE_INT // 4):
    buffer.popleft()

But this is optional.
What must stay consistent

Even if overlap differs, these must match between train and infer:

    same WINDOW_SIZE

    same channels used (Ch0-3 Act)

    same feature preset (baseline vs baseline_plus_wamp)

So yes: just training overlap is enough and is a good next step.

You said:
Best validation accuracy:       94.87%
Saved model:                   semg_mlp_best.pth
Saved encoder:                 label_encoder_mlp_best.pkl

============================================================
FINAL EVALUATION ON UNSEEN DATA (RUN ONCE)
============================================================
semg_infer_mlp.py --mode offline --csv ..\sutd_bmi_safety_data\unseen\*.csv --feature_preset baseline_plus_wamp
Feature preset: baseline_plus_wamp
Model input_dim: 28

Metrics per class:
Class: extension
  Accuracy:  1.0000
Class: flexion
  Accuracy:  0.7778
Class: grasp
  Accuracy:  0.4444
Class: left
  Accuracy:  0.6667
Class: nothing
  Accuracy:  1.0000
Class: open
  Accuracy:  0.8889
Class: pronation
  Accuracy:  0.6667
Class: right
  Accuracy:  0.6667
Class: supination
  Accuracy:  0.8889

Overall Metrics:
  Accuracy:  0.7778

Predictions saved to: predictions_20260225_004417.csv

Confusion Matrix:
(Rows: Actual, Columns: Predicted)
            extension  flexion  grasp  left  ...  open  pronation  right  supination
extension           9        0      0     0  ...     0          0      0           0
flexion             0        7      1     0  ...     0          0      0           0
grasp               0        1      4     2  ...     0          1      0           1
left                0        0      0     6  ...     0          1      0           2
nothing             0        0      0     0  ...     0          0      0           0
open                0        0      0     0  ...     8          0      1           0
pronation           0        0      0     3  ...     0          6      0           0
right               0        0      0     0  ...     2          0      6           1
supination          0        0      0     0  ...     0          0      1           8

[9 rows x 9 columns]

Confusion matrix saved to: confusion_matrix_20260225_004417.csv
honeslty every preset has barely any idfferent 

ChatGPT said:

Yeah — and that’s actually a useful finding.

If changing presets (baseline vs +WAMP) only moves things a little, it usually means the bottleneck is not just features anymore. It’s more likely:

    data quality / labeling

    class overlap in physiology (gestures genuinely look similar)

    small dataset size

    session shift (train vs unseen conditions differ)

Your confusion matrix gives strong clues.
What your confusion matrix is saying

The errors are structured (not random), which is good:

    grasp gets confused with flexion, left, pronation, supination

    pronation gets confused with left

    left gets confused with pronation and supination

    right gets confused with open and supination

    open gets confused with right

So your model is learning something, but some classes are biomechanically close and the current setup (4 channels + simple time-domain features + MLP) doesn’t separate them strongly enough.
What to do next (high impact, not feature preset tweaks)
1) Improve data split and labeling first

This is the biggest thing.
A) Remove transition windows

A lot of EMG mistakes come from windows that include gesture transitions (e.g., moving into grasp from open).

Even if your training script keeps only single-label windows, your unseen inference labels each whole file as one class, and the file may contain:

    startup movement

    settling period

    drift

    release movement

That creates “fake mistakes.”
Quick fix for unseen evaluation

In offline_predict, skip the first and last part of each file (e.g., 0.5s–1s) before evaluating.

If WINDOW_SIZE=100 (~1s), try:

    skip first 50 rows

    skip last 50 rows

That alone can noticeably improve apparent accuracy.
2) Add overlap in training (if not already)

This helps more than tiny feature changes.

Use:

STEP_SIZE = int(WINDOW_SIZE // 4)

This increases training samples and robustness.
3) Add class-weighted loss

Your hard classes are clearly underperforming. Weighted loss helps the model pay more attention to them.

This is probably more impactful now than adding one more feature.
4) Normalize each window per channel

This can help session-to-session drift (electrode pressure, skin contact, effort level).

Inside extract_features, before computing features for each channel:

ch = (ch - np.mean(ch)) / (np.std(ch) + 1e-8)

This is a very common EMG trick.

You can make it a preset option too (normalize_per_channel=True) so it stays reversible.
5) Consider a hierarchical label strategy (later)

Some of your confusions are “direction family” confusions:

    left / right / pronation / supination

    open / grasp / flexion

A single flat classifier may struggle. A two-stage classifier can help:

    coarse group (rest vs hand-shape vs rotation)

    specific gesture within group

Not necessary yet, but it fits your confusion pattern.
6) Collect targeted data (this is probably the real answer now)

At this point, yes — targeted recollection will likely help more than feature presets.

Focus on the confused pairs:

    grasp vs flexion

    left vs pronation/supination

    right vs open/supination

How to collect better data

For each weak gesture:

    3–5 separate recordings (different times)

    hold the gesture steady for a few seconds

    include varied intensity (light/medium/strong)

    keep transitions separate (don’t start recording while moving into pose)

This improves separability much more than squeezing more handcrafted features.
Most likely reason your presets barely differ

Because your baseline features are already pretty solid:

    MAV, RMS, WL, VAR, ZC, SSC

That’s already a strong classic EMG set. Adding WAMP helps a bit (as you saw), but it won’t magically fix:

    noisy labels

    transition windows

    inter-session variation

    limited channels

So the small gain is normal.
Best next experiment (practical and worth it)

If you want a meaningful next jump, do this combo:
✅ Keep baseline_plus_wamp
✅ Add overlap in training
✅ Add class-weighted loss
✅ Add per-channel normalization before features
✅ Skip start/end rows in unseen evaluation

That’s the highest ROI before changing architecture.

If you want, I can give you the exact copy-paste patches for:

    training overlap

    weighted loss

    per-channel normalization (toggle-able preset)

    skipping transition rows in offline evaluation

and we can do it one step at a time.

You said:
so i should do weighted loss and collecting more samples for worser doing samples?

ChatGPT said:

Yes — that’s the right next move. ✅
Priority order (best ROI)

    Add class-weighted loss (easy, immediate)

    Collect more samples for weak classes (biggest real improvement)

    (Optional but helpful) overlap in training

    (Optional) per-channel normalization

Why weighted loss helps

Your model is doing well overall, but it underperforms on:

    grasp

    left/right/pronation (some of them depending on run)

Weighted loss tells training:

    “Mistakes on these classes matter more.”

That helps when:

    classes have fewer windows

    or some classes are harder to separate

Why more samples helps even more

Your confusion matrix shows consistent confusions, not random ones.

That usually means the model needs more examples of the actual variation for those gestures:

    different force

    slight wrist angle changes

    different session/electrode contact

Especially for EMG, data collection quality matters a lot.
What to collect (targeted)

Focus on the weak/confused ones:

    grasp

    left

    right

    pronation
    (and maybe flexion if still unstable)

Tips while collecting

    Record steady holds (not transitions)

    Do multiple sessions (not all in one sitting)

    Vary intensity (light/medium/strong)

    Keep labels clean

This will help more than just adding more features.
Do both together

Best combo:

    weighted loss + more targeted samples

That gives you:

    immediate improvement from training objective

    long-term improvement from better data

If you want, I can give you the exact weighted loss code patch for your training script (just a few lines in the right spot).

You said:
show me the code for weighted loss

ChatGPT said:

Yep — here’s the exact patch.
Replace this in your training script

You currently have:

criterion = nn.CrossEntropyLoss()

Replace it with this after y_train is created and after device is defined:

# Class-weighted loss (helps weaker / underrepresented classes)
class_counts = np.bincount(y_train, minlength=len(le.classes_))

# Avoid divide-by-zero just in case
class_counts = np.maximum(class_counts, 1)

# Inverse-frequency style weights
class_weights = class_counts.sum() / (len(class_counts) * class_counts)

print("\nClass counts (train):")
for i, cls in enumerate(le.classes_):
    print(f"  {cls}: {class_counts[i]}")

print("\nClass weights:")
for i, cls in enumerate(le.classes_):
    print(f"  {cls}: {class_weights[i]:.4f}")

class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

Where to put it exactly

Put it in your script around here:

model = SEMGMLP(n_classes, input_dim)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 3. Training
# <-- put weighted loss code here
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

What this does

    classes with fewer train windows get higher weight

    classes with many windows get lower weight

    the loss penalizes mistakes on weak classes more

Optional (safer version if weights become too extreme)

Sometimes inverse-frequency weights can be too aggressive.
You can soften them using square root:

class_weights = np.sqrt(class_counts.sum() / (len(class_counts) * class_counts))

If training becomes unstable, use this softened version.
Important

Keep everything else the same first (same feature preset, same windowing) so you can see whether weighted loss helped.

If you want, I can also give you a version that combines:

    weighted loss

    overlap training

    prints per-class train counts
    all in one clean patch.


You said:
but all the classes have the same number of windows

ChatGPT said:

Yep — then weighted loss probably won’t help much (or at all). If class counts are equal, standard cross-entropy is already fair by count.

So in your case, the issue is more likely:

    classes are intrinsically harder / more similar (grasp vs flexion, left vs pronation, etc.)

    data quality / transitions

    session variation (electrode placement, effort, noise)

What to do instead (best next steps)
1) Collect more samples for the hard/confused classes

Even if counts are equal, you still want more diversity for the weaker classes:

    grasp

    left/right

    pronation (and maybe flexion)

Not just more quantity — more variation:

    different force levels

    different wrist angles

    different sessions

That’s the biggest likely gain.
2) Remove transition-heavy windows from evaluation/training

This matters a lot for EMG.

If a “grasp” file includes:

    moving into grasp

    unstable settling

    relaxing out of grasp

then some windows won’t look like clean grasp.

A quick improvement is to:

    skip the first and last part of each file during offline evaluation

3) Use overlap in training

Even with balanced classes, overlap helps create more robust windows.

STEP_SIZE = int(WINDOW_SIZE // 4)

This is a good next change.
4) Normalize each channel per window before features

This helps reduce session/electrode variation.

Inside extract_features (or per-channel before computing features):

ch = (ch - np.mean(ch)) / (np.std(ch) + 1e-8)

This can help unseen performance more than another feature tweak.
5) If you want a “hard-class” loss, use Focal Loss (optional)

Since your problem is hard examples, not class imbalance, Focal Loss is a better fit than weighted CE.

Focal loss downweights easy examples and focuses more on mistakes/hard ones.

I can give you a drop-in PyTorch focal loss if you want.
Bottom line

Since your classes are balanced:

    ❌ weighted loss is not the priority

    ✅ focus on better data for weak classes, overlap windows, and cleaner windows

    ✅ normalization is a strong next experiment

If you want, I can give you the exact code patch for:

    overlap training, and

    skipping first/last rows in offline evaluation (very easy and useful).


You said:
semg_inference_mlp.py.pdf
PDF
this are my resulst right now

ChatGPT said:

These are actually good, useful results — and they tell a clear story.

From your slides/results:

    baseline unseen is around 0.74–0.80

    adding WAMP helps in some runs (especially grasp/flexion/pronation)

    75% overlap sometimes helps, but not consistently

    your model is still most sensitive on a few gestures (grasp / flexion / rotation-type classes)

What your results mean

You’re at the stage where:

    the feature engineering is already decent

    gains from presets are incremental

    the main bottleneck is likely data quality / session variation / class ambiguity, not just “missing one feature”

That’s normal for sEMG.
Why overlap didn’t always help

Overlap gives more training windows, but they’re highly correlated (same recording, just shifted).
So it can:

    improve robustness a bit

    but also make validation look better without improving unseen much

Your page showing sample count jump (~192 → ~745) is a nice improvement in volume, but it doesn’t create new signal diversity by itself.
What I’d conclude from your current experiments
✅ Keep

    feature preset system (very good design choice)

    WAMP option (it gave real gains in some runs)

    iterative search by validation (correct methodology now)

⚠️ Don’t over-focus on

    squeezing tiny gains from more handcrafted feature combinations
    You’ve already shown the improvements are small/inconsistent.

Best next steps (practical)
1) Collect more data for weak/confused gestures

Yes — this is the biggest next improvement.
Focus on:

    grasp

    flexion

    pronation / left / right / supination confusion set

Collect:

    multiple sessions

    steady holds

    different intensities

    slightly different wrist angles

2) Clean evaluation windows

In unseen evaluation, skip the first/last part of each file (transition/noisy segments).
This often improves fairness a lot for EMG.
3) Try per-window channel normalization

This helps session-to-session drift and electrode placement differences.
4) Keep overlap only if it helps unseen

Since your results are mixed, use overlap only if it improves unseen, not just validation.
Overall

Your pipeline is already in a strong state:

    modular features

    reproducible iterative testing

    confusion matrix tracking

    unseen evaluation

That’s honestly solid work. The next meaningful jump will probably come more from better targeted data than from another feature tweak.

If you want, I can help you design a targeted recollection plan (how many files per gesture, durations, naming, and a clean protocol) so your next data collection round directly improves the weak classes.

You said:
C:\Users\leewe\Desktop\semg\sutd_bmi_safety\src\testbed\sutd_bmi_safety_data\additional

i've collected additional grasp, flexion and pronation data, how do i use it in the training

ChatGPT said:

Nice — that’s exactly the right classes to target.
Use the new data in training (simple + safe)
Best approach

Append the new CSVs into your training dataset (combined.csv) and retrain.

But do not mix them into your unseen test folder (keep unseen separate).
Option 1 (recommended): Create a new combined training CSV

Make a script (or one-time notebook) that:

    loads your original combined.csv

    loads all CSVs in:
    ...\sutd_bmi_safety_data\additional

    concatenates them

    saves as something like combined_v2.csv

Example script

import os
import glob
import pandas as pd

base_path = r"..\sutd_bmi_safety_data\combined.csv"
additional_folder = r"..\sutd_bmi_safety_data\additional"
output_path = r"..\sutd_bmi_safety_data\combined_v2.csv"

# Load original
df_main = pd.read_csv(base_path)

# Load additional CSVs
files = glob.glob(os.path.join(additional_folder, "*.csv"))
print("Found additional files:", len(files))

dfs = [df_main]
for f in files:
    df = pd.read_csv(f)
    print(f, df.shape)
    dfs.append(df)

# Combine
df_all = pd.concat(dfs, ignore_index=True)

# Save
df_all.to_csv(output_path, index=False)
print("Saved:", output_path, "Shape:", df_all.shape)

Then train with:

python semg_train_mlp.py --data ..\sutd_bmi_safety_data\combined_v2.csv --feature_preset baseline_plus_wamp

Important checks before concatenating

Make sure the additional CSVs have the same columns as your training file:

    Ch0 Act

    Ch1 Act

    Ch2 Act

    Ch3 Act

    Action

If the new files are single-gesture files but don’t have an Action column, you need to add it first.
If your additional files don’t have Action column

If filenames contain the label (like grasp_01.csv, flexion_02.csv), you can auto-add it.
Example

import os
import glob
import pandas as pd

additional_folder = r"..\sutd_bmi_safety_data\additional"
files = glob.glob(os.path.join(additional_folder, "*.csv"))

label_names = ["grasp", "flexion", "pronation"]  # extend if needed
all_new = []

for f in files:
    df = pd.read_csv(f)
    fname = os.path.basename(f).lower()

    # infer label from filename
    label = None
    for cls in label_names:
        if cls in fname:
            label = cls
            break

    if label is None:
        print("Skipping (no label inferred):", fname)
        continue

    if "Action" not in df.columns:
        df["Action"] = label

    all_new.append(df)
    print(f"Loaded {fname} as label={label}, shape={df.shape}")

df_new = pd.concat(all_new, ignore_index=True)
df_new.to_csv(r"..\sutd_bmi_safety_data\additional_labeled.csv", index=False)
print("Saved labeled additional data.")

Then append that to combined.csv.
Option 2: Keep files separate and train from a folder (cleaner later)

You can modify your training script to read all CSVs from multiple folders and concatenate internally, but for now Option 1 is faster and less risky.
Very important: avoid data leakage

If your “additional” data is from the same session style as unseen and you also evaluate on similar files, that’s okay — as long as the exact files are not in the unseen folder.

Rule:

    train/val data = combined_v2.csv

    unseen test data = separate folder only (never included in combined)

After adding data, do this

    Retrain using combined_v2.csv

    Print windows-per-class again (you should see grasp/flexion/pronation increase)

    Run your search script with baseline_plus_wamp

    Compare unseen confusion matrix

You should expect the biggest gains in:

    grasp

    flexion

    pronation

Extra tip (helpful)

Since you added only a few classes, your class counts may become imbalanced (that’s okay).
At that point, weighted loss may actually become useful.

If you want, I can give you a single merge script that:

    reads original + additional

    auto-labels missing Action

    validates columns

    prints class counts before/after

    writes combined_v2.csv in one go.


You said:
how to run search best.py again

ChatGPT said:

From your project folder (where semg_search_best.py is), run it like this.
Baseline + WAMP (recommended)

python semg_search_best.py --data ..\sutd_bmi_safety_data\combined_v2.csv --feature_preset baseline_plus_wamp

Baseline only

python semg_search_best.py --data ..\sutd_bmi_safety_data\combined_v2.csv --feature_preset baseline

If you want fewer/more iterations

Default is 20. You can change it:

python semg_search_best.py --data ..\sutd_bmi_safety_data\combined_v2.csv --feature_preset baseline_plus_wamp --iterations 10

If your unseen folder path is different

You can pass it explicitly too:

python semg_search_best.py --data ..\sutd_bmi_safety_data\combined_v2.csv --feature_preset baseline_plus_wamp --unseen_csv "..\sutd_bmi_safety_data\unseen\*.csv"

Before running

Make sure:

    combined_v2.csv exists

    semg_train_mlp.py and semg_infer_mlp.py both support --feature_preset

    old model files can be deleted (the script handles this)

If you want, paste the output after 1 run and I’ll help interpret whether the added grasp/flexion/pronation data helped.

You said:
C:\Users\leewe\Desktop\semg\sutd_bmi_safety\src\testbed\sutd_bmi_safety_data\baseline

heres my file path

ChatGPT said:

Got it — your folder is:

C:\Users\leewe\Desktop\semg\sutd_bmi_safety\src\testbed\sutd_bmi_safety_data\baseline

So if your new merged training file is inside that folder (for example combined_v2.csv), run from your testbed\ml folder like this:

python semg_search_best.py --data ..\sutd_bmi_safety_data\baseline\combined_v2.csv --feature_preset baseline_plus_wamp

If your file is still named combined.csv

Use:

python semg_search_best.py --data ..\sutd_bmi_safety_data\baseline\combined.csv --feature_preset baseline_plus_wamp

If you want to be extra safe, use full path

(Windows full path works too)

python semg_search_best.py --data "C:\Users\leewe\Desktop\semg\sutd_bmi_safety\src\testbed\sutd_bmi_safety_data\baseline\combined_v2.csv" --feature_preset baseline_plus_wamp

Make sure you run it from the correct folder

Open terminal in:

C:\Users\leewe\Desktop\semg\sutd_bmi_safety\src\testbed\ml

Then run the command.

If you run it from somewhere else, relative paths (..\...) may break.
Optional: pass unseen path explicitly too

If your unseen folder is also under sutd_bmi_safety_data, do:

python semg_search_best.py --data ..\sutd_bmi_safety_data\baseline\combined_v2.csv --unseen_csv "..\sutd_bmi_safety_data\unseen\*.csv" --feature_preset baseline_plus_wamp

If you want, send me the exact filename inside baseline (like combined.csv / combined_v2.csv) and I’ll give you the exact one-line command with no guessing.

You said:
i added 5 more datasts for those 3 and here are the resulst
Best validation accuracy:       84.00%
Saved model:                   semg_mlp_best.pth
Saved encoder:                 label_encoder_mlp_best.pkl

============================================================
FINAL EVALUATION ON UNSEEN DATA (RUN ONCE)
============================================================
semg_infer_mlp.py --mode offline --csv ..\sutd_bmi_safety_data\unseen\*.csv --feature_preset baseline
Feature preset: baseline
Model input_dim: 24

Metrics per class:
Class: extension
  Accuracy:  1.0000
Class: flexion
  Accuracy:  0.3333
Class: grasp
  Accuracy:  0.4444
Class: left
  Accuracy:  0.6667
Class: nothing
  Accuracy:  1.0000
Class: open
  Accuracy:  0.7778
Class: pronation
  Accuracy:  1.0000
Class: right
  Accuracy:  0.6667
Class: supination
  Accuracy:  0.8889

Overall Metrics:
  Accuracy:  0.7531

Predictions saved to: predictions_20260225_212001.csv

Confusion Matrix:
(Rows: Actual, Columns: Predicted)
            extension  flexion  grasp  left  ...  open  pronation  right  supination
extension           9        0      0     0  ...     0          0      0           0
flexion             0        3      4     0  ...     0          0      0           0
grasp               0        0      4     2  ...     0          3      0           0
left                0        0      0     6  ...     0          3      0           0
nothing             0        0      0     0  ...     0          0      0           0
open                1        0      0     0  ...     7          0      1           0
pronation           0        0      0     0  ...     0          9      0           0
right               0        0      0     0  ...     3          0      6           0
supination          0        0      0     0  ...     0          0      1           8

[9 rows x 9 columns]

ChatGPT said:

This is a good result to look at carefully — and it explains a lot.
What changed

After adding more data for grasp / flexion / pronation:

    Validation dropped to 84% (from ~95–100%)

    Unseen stayed around ~75% (0.7531)

That usually means your training set is now harder / more realistic, which is actually a good sign.
Why validation dropped

Because you added more variation (good thing):

    more signal diversity

    maybe noisier examples

    more realistic execution differences

So the model can’t “memorize” as easily anymore.
This makes validation accuracy more honest.

A lower validation score is not bad if the data got more realistic.
What the confusion matrix says now

This is the key part.
1) Flexion is still the biggest problem

flexion -> grasp is the main failure:

    flexion accuracy = 0.3333

    4 of 9 flexion windows predicted as grasp

So flexion/grasp are still overlapping heavily.
2) Grasp improved a bit, but now confuses with pronation/left

grasp -> pronation and grasp -> left

This suggests the new grasp samples may have introduced more variability, but not necessarily “cleaner separation.”
3) Pronation became excellent

pronation = 1.0000 ✅
That means your added pronation data helped a lot.

So your targeted recollection worked for pronation.
Big takeaway

Your added data helped one class a lot (pronation), but flexion vs grasp is still the main unresolved pair.

That means the next step should be targeted specifically for flexion/grasp separation, not just “more general data.”
What I’d do next (practical)
1) Use baseline_plus_wamp again on the new dataset

The run you showed is baseline (24 features).

Since WAMP helped flexion/grasp before, rerun with:

python semg_search_best.py --data ..\sutd_bmi_safety_data\baseline\combined_v2.csv --feature_preset baseline_plus_wamp

That’s the fairest comparison after adding data.
2) Collect cleaner flexion/grasp data specifically

Not just more samples — make them easier to distinguish in collection:
For each file:

    start recording only after gesture is stable

    hold 2–3 seconds steady

    stop before relaxing

    avoid transitions in the recorded portion

For flexion vs grasp:

Try to make the movement intent very distinct:

    flexion: wrist-focused movement, relaxed fingers

    grasp: finger closure grip, minimal wrist movement (if that’s your label definition)

Right now, your EMG may be capturing mixed patterns.
3) Evaluate only the steady middle of unseen files

This is a big one.

Your offline evaluator labels every prediction in the whole file as the gesture, but the file likely includes:

    start transition

    end transition

Those windows are often what get misclassified.
Easy fix

In offline_predict, skip the first and last part of each file. For example:

start_idx = 50
end_idx = len(df) - 50
for i in range(start_idx, end_idx):
    ...

This makes evaluation more fair and often improves flexion/grasp a lot.
4) Keep overlap in training (if not already)

If you haven’t enabled it yet:

STEP_SIZE = int(WINDOW_SIZE // 4)

This can help robustness, especially with added data.
5) (Optional) Add per-channel normalization

This helps if your new datasets were collected in slightly different conditions.

Inside feature extraction, per channel:

ch = (ch - np.mean(ch)) / (np.std(ch) + 1e-8)

This often helps unseen consistency.
Why “more data” didn’t instantly boost overall accuracy

Because more data can do two things:

    improve generalization ✅

    expose ambiguity / noise ✅

You likely improved pronation a lot, but also introduced more variability in flexion/grasp that the current features + MLP still can’t fully separate.

That’s not failure — it’s progress. You now know exactly where the bottleneck is.
What I’d recommend next run

Do this combo on your new merged dataset:

    baseline_plus_wamp

    overlap training (WINDOW_SIZE // 4)

    skip first/last rows in unseen evaluation

That’s the highest-value next test.

If you want, I can give you the exact copy-paste code patch for the “skip first/last rows” change in your offline_predict() right now.

You said:
do flexion and grasp use different muscles in the forearm

ChatGPT said:

Yes — overlapping, but not identical.
Short version

    Grasp and flexion both can activate forearm muscles, so EMG signals can look similar.

    But they usually emphasize different muscle groups depending on what you mean by “flexion.”

If by “grasp” you mean gripping/closing the hand

This mainly uses the finger flexors in the forearm, especially:

    Flexor digitorum superficialis

    Flexor digitorum profundus

    plus thumb-related muscles (some in forearm, some in hand)

If by “flexion” you mean wrist flexion

This mainly uses wrist flexors, like:

    Flexor carpi radialis (FCR)

    Flexor carpi ulnaris (FCU)

    Palmaris longus (if present)

Why your model confuses them

Because in real movement:

    people often co-contract (activate both)

    gripping often includes a little wrist flexion

    wrist flexion may also tighten finger flexors slightly

    electrode placement can pick up mixed signals from nearby muscles

So even though the target muscles differ, your 4-channel forearm EMG can still capture overlapping patterns.
Practical takeaway for your dataset

To make them easier to separate:
For grasp recordings

    keep wrist neutral

    focus on finger closing / grip

    avoid bending the wrist

For flexion recordings

    flex the wrist

    keep fingers relaxed/open if possible

    avoid gripping during the motion

That recording style difference can help your model a lot.

If you want, I can also suggest a simple gesture protocol (how to position wrist/fingers for each class) so your future recordings are more separable.

You said:
would using cnn be different from mlp

ChatGPT said:

Yes — it can be meaningfully different, and in your case it’s worth trying.
Main difference
MLP (what you’re using now)

    uses handcrafted features (MAV, RMS, WL, etc.)

    model only sees the summary numbers

    simpler, fast, works well with small datasets

CNN (1D CNN for sEMG)

    uses the raw time-series window directly

    learns patterns automatically (shape, spikes, timing, local signal structure)

    can capture subtle differences that handcrafted features miss

Why CNN may help your flexion vs grasp problem

Your current issue is likely subtle signal-shape differences and co-activation patterns.

A CNN can learn:

    local waveform patterns

    timing/shape differences across channels

    short bursts / transitions

    channel interactions (depending on architecture)

That’s something MLP on summary features may lose.
But CNN is not automatically better

CNN helps if:

    you have enough data (or overlap windows)

    labels are clean

    train/infer preprocessing is consistent

CNN can perform worse than MLP if:

    dataset is too small

    labels are noisy

    classes are heavily overlapping physiologically

In your setup, I’d say: yes, try CNN

You already have:

    a working pipeline

    unseen evaluation

    confusion matrix

    targeted data collection

That’s the perfect point to test a CNN baseline fairly.
Very important: your current SEMGCNN is set to in_channels=2

But your data uses 4 channels (Ch0 Act to Ch3 Act), so if you use your current CNN class as-is, it won’t match.

You’d need:

self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=5, padding=2)

What changes if you use CNN
For MLP (current)

Input shape per sample:

    (24,) or (28,) features

For CNN

Input shape per sample:

    (4, WINDOW_SIZE) raw signal

So training code changes:

    don’t call extract_features()

    feed raw windows directly into dataset/model

Suggested strategy (best)

Don’t replace MLP yet.
Run CNN as a separate experiment and compare unseen accuracy.
Compare:

    MLP + baseline_plus_wamp

    CNN on raw windows

Same train data, same unseen data.

That gives a fair answer.
What I expect

    CNN might improve the hard confusions (flexion/grasp), especially with more data

    But if labels are noisy (transitions), CNN may still struggle

So CNN is promising, but data quality still matters most.
Bottom line

Yes — CNN is different from MLP and can capture signal patterns your handcrafted features miss.
Given your confusion matrix, it’s a very reasonable next experiment.

If you want, I can give you a drop-in semg_train_cnn.py and the matching offline inference CNN script using your current pipeline structure.

You said:
whats the difference between cnn and mlp 

ChatGPT said:

A CNN and an MLP are both neural networks, but they process the data differently.
MLP

    Input: a flat vector (e.g., your handcrafted features like MAV, RMS, etc.)

    How it works: every neuron connects to all inputs (“fully connected”)

    Learns: relationships between the summary values

    Good for: small datasets, tabular/feature-based input

In your case, MLP sees:

    MAV, RMS, WL, ... for each channel
    It does not see the raw waveform shape.

CNN

    Input: structured data (for sEMG, the raw signal window: channels × time)

    How it works: uses convolution filters that slide across the signal

    Learns: local patterns (spikes, bursts, timing shapes) automatically

    Good for: time series, images, signals

In your case, CNN sees:

    the full raw sEMG window (e.g., 4 × 100 samples)

    so it can learn shape/timing patterns that handcrafted features may miss

Simple analogy

    MLP = you give it a summary report

    CNN = you give it the raw signal and let it learn what matters

Why this matters for your project

Your hard classes (like flexion vs grasp) may have:

    similar averages (MAV/RMS)

    but slightly different waveform patterns over time

A CNN may detect those differences better than an MLP.
Tradeoff

    MLP: simpler, faster, easier, needs good features

    CNN: more powerful for raw signals, but needs more data and cleaner labels

So both are valid — they just learn from different representations of your sEMG.

You said:
what is wamp ? 

ChatGPT said:

WAMP = Willison Amplitude.

It’s a classic sEMG feature that counts how often the signal changes “enough” between consecutive samples.
Intuition

It measures how active / twitchy / bursty the EMG signal is.

Instead of looking at average amplitude, WAMP asks:

    “How many times did the signal jump by more than a threshold?”

Formula idea

For a signal x:

    look at |x[i] - x[i-1]|

    if it’s greater than a threshold, count it

    sum over the whole window

So WAMP is basically a count of significant changes.
Why it helps in EMG

Two gestures can have similar:

    MAV

    RMS

but different activation dynamics (how jagged/active the signal is).

WAMP can help separate gestures like:

    flexion vs grasp

    open vs rest
    when the timing/activity pattern differs.

In your code

Your WAMP looked like this:

def _wamp(x, threshold=0.02):
    return np.sum(np.abs(np.diff(x)) > threshold)

That means:

    np.diff(x) = consecutive sample changes

    np.abs(...) > threshold = significant jump?

    np.sum(...) = count of jumps

Threshold matters

The threshold controls sensitivity:

    too low → counts noise

    too high → misses real muscle activity

So 0.02 is a tunable hyperparameter (depends on your signal scaling).
Quick comparison with other features

    MAV/RMS = “how strong is the signal overall?”

    WL = “how much total movement in the signal?”

    ZC/SSC = “how often direction/slope changes?”

    WAMP = “how often the signal makes a meaningful jump?”

It’s a nice complement to your existing feature set.

You said:
    Mean Absolute Value (MAV): Measures the average amplitude of the signal and reflects overall muscle activation intensity. 

[Equation] 

 

    Root Mean Square (RMS): Represents the power of the sEMG signal and is often correlated with muscle contraction strength. 

[Equation] 

 

    Waveform Length (WL): Captures the cumulative signal changes within a window, providing a measure of muscle fatigue and activity complexity. 

[Equation] 

 

 

    Variance (VAR): Indicates the spread of signal energy, useful for distinguishing between static and dynamic muscle states. 

[Equation] 

 

    Zero Crossing (ZC): Counts how often the signal crosses zero, which is related to muscle fibre firing frequency. 

[Equation] 

 

    Slope Sign Changes (SSC): Counts how many times the slope of the signal changes sign, representing signal complexity and contraction transitions. 

[Equation] 

give me short wamp  description and equation

ChatGPT said:

Willison Amplitude (WAMP): Counts the number of times the absolute difference between two consecutive sEMG samples exceeds a predefined threshold, capturing significant signal changes and muscle activation dynamics.

Equation:
WAMP=∑i=1N−1f(∣xi+1−xi∣)
WAMP=i=1∑N−1​f(∣xi+1​−xi​∣)

where
f(u)={1,u>T0,u≤T
f(u)={1,0,​u>Tu≤T​

    xixi​: sEMG sample at time ii

    NN: number of samples in the window

    TT: threshold (to suppress noise)


You said:
Willison Amplitude (WAMP): Counts the number of times the absolute difference between two consecutive sEMG samples exceeds a predefined threshold, capturing significant signal changes and muscle activation dynamics.

what does this even mean

ChatGPT said:

It means:

WAMP counts how many “big jumps” happen in the signal within one window.
Super simple version

Imagine your sEMG signal is a line going up and down.

WAMP checks each pair of neighboring points:

    if the change is small → ignore it

    if the change is big enough (above a threshold) → count 1

Then it adds them up.

So:

    higher WAMP = signal is changing a lot (more muscle activity / more bursts)

    lower WAMP = signal is smoother or quieter

Tiny example

Signal window:
[0.10, 0.12, 0.30, 0.32, 0.50]

Differences between consecutive points:

    |0.12 - 0.10| = 0.02

    |0.30 - 0.12| = 0.18

    |0.32 - 0.30| = 0.02

    |0.50 - 0.32| = 0.18

If threshold = 0.05, then:

    0.02 → no

    0.18 → yes

    0.02 → no

    0.18 → yes

So WAMP = 2
Why threshold is there

To ignore tiny changes caused by noise.

Without a threshold, even small jitter would be counted.
In EMG terms

WAMP is like asking:

    “How often is the muscle signal making meaningful changes, not just tiny noise wiggles?”

That’s why it can help distinguish gestures that have similar average amplitude but different signal dynamics.

You said:
import numpy as np
import torch
from collections import deque
import socketio
import time
from semg_model import SEMGCNN, WINDOW_SIZE
import joblib

# Load label encoder and model
le = joblib.load("label_encoder_cnn.pkl")
n_classes = len(le.classes_)
model = SEMGCNN(n_classes=n_classes)
model.load_state_dict(torch.load("semg_cnn.pth", map_location="cpu"))
model.eval()

# Initialize buffer
buffer = deque(maxlen=WINDOW_SIZE)

def process_and_predict(data):
    try:
        raw = np.array(
            [
                data["ch0"]["a"],
                # data["ch0"]["e"],
                data["ch1"]["a"],
                # data["ch1"]["e"],
                data["ch2"]["a"],
                # data["ch2"]["e"],
                data["ch3"]["a"],
                # data["ch3"]["e"],
            ]
        )
        buffer.append(raw)
        if len(buffer) == WINDOW_SIZE:
            window = np.stack(buffer, axis=1)

            # Normalize per channel
            norm_signal = (window - window.mean(axis=1, keepdims=True)) / (window.std(axis=1, keepdims=True) + 1e-6)
            tensor = torch.tensor(norm_signal, dtype=torch.float32).unsqueeze(0)  # (1, 2, 600)

            with torch.no_grad():
                out = model(tensor)
                pred = out.argmax(dim=1).item()
                buffer.clear()
                return le.inverse_transform([pred])[0]
    except Exception as e:
        print(f"Inference error: {e}")
    return None

# SocketIO setup
sio = socketio.Client()

@sio.event
def connect():
    print("Connected to Socket.IO server.")

@sio.on("adc_data")
def semg_data(data):
    prediction = process_and_predict(data)
    if prediction:
        print(f"Detection: {prediction}")

if __name__ == "__main__":
    sio.connect("http://localhost:3000", transports=["websocket"], namespaces=["/"])
    try:
        print("waiting...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        sio.disconnect()

how does my cnn code work

ChatGPT said:

It’s a real-time sEMG gesture detector using a CNN.
It listens to incoming sensor data from Socket.IO, builds a fixed-size window, runs the CNN, and prints the predicted gesture.
How it works (step by step)
1) Load the trained CNN + label encoder

le = joblib.load("label_encoder_cnn.pkl")
model = SEMGCNN(n_classes=n_classes)
model.load_state_dict(torch.load("semg_cnn.pth", map_location="cpu"))
model.eval()

    label_encoder_cnn.pkl converts numeric class IDs back to gesture names

    semg_cnn.pth contains your trained CNN weights

    model.eval() puts the model in inference mode

2) Create a sliding buffer for incoming data

buffer = deque(maxlen=WINDOW_SIZE)

    This stores the latest WINDOW_SIZE samples

    Once it fills up, you have one complete input window for prediction

3) Each incoming message (adc_data) contains one sample

In process_and_predict(data), you extract the 4 active channels:

raw = np.array([
    data["ch0"]["a"],
    data["ch1"]["a"],
    data["ch2"]["a"],
    data["ch3"]["a"],
])

So raw is one time-step with 4 channel values.

Then:

buffer.append(raw)

This adds the sample to the rolling buffer.
4) When buffer is full, form the CNN input window

window = np.stack(buffer, axis=1)

This gives shape:

    buffer stores WINDOW_SIZE items, each of shape (4,)

    stacking with axis=1 gives window.shape = (4, WINDOW_SIZE)

That is exactly what a 1D CNN expects:

    4 channels

    sequence length = WINDOW_SIZE

5) Normalize per channel

norm_signal = (window - window.mean(axis=1, keepdims=True)) / (window.std(axis=1, keepdims=True) + 1e-6)

This standardizes each channel independently:

    mean → 0

    std → 1

Why this helps:

    reduces session/electrode amplitude differences

    makes inference more stable

6) Convert to PyTorch tensor and add batch dimension

tensor = torch.tensor(norm_signal, dtype=torch.float32).unsqueeze(0)

Now shape becomes:

    (1, 4, WINDOW_SIZE)

Where:

    1 = batch size

    4 = channels

    WINDOW_SIZE = time length

⚠️ Your comment says (1, 2, 600) — that comment is outdated.
It should be something like:

# shape: (1, 4, WINDOW_SIZE)

7) Run CNN inference

with torch.no_grad():
    out = model(tensor)
    pred = out.argmax(dim=1).item()

    out is the model’s logits for each class

    argmax picks the class with highest score

    pred is the numeric class index

Then:

return le.inverse_transform([pred])[0]

converts class ID → label string (e.g., "grasp")
8) Clear buffer after each prediction

buffer.clear()

This means your inference is non-overlapping windows:

    predict once per full window

    start fresh after each prediction

So if WINDOW_SIZE = 100, you predict every 100 samples.
9) Socket.IO event loop

@sio.on("adc_data")
def semg_data(data):
    prediction = process_and_predict(data)
    if prediction:
        print(f"Detection: {prediction}")

This runs every time the server sends new sensor data.

Main block:

sio.connect(...)
while True:
    time.sleep(1)

keeps the client connected and listening.
Important thing to check (very important)

Your CNN model in semg_model.py must match the input:
You are feeding 4 channels

So CNN must be:

nn.Conv1d(in_channels=4, ...)

If your SEMGCNN still has in_channels=2, this code will fail (or already failed before).
One more important thing: training must match inference

Since you normalize per channel in inference, you should also normalize the same way in CNN training.

If training didn’t normalize, inference normalization might hurt performance.
Summary

Your CNN inference code:

    receives live 4-channel sEMG samples

    stores them in a fixed window

    normalizes the window

    feeds raw signal into CNN

    prints predicted gesture

    resets buffer for the next prediction

It’s a clean real-time pipeline 👌

If you want, I can also review your CNN training code to make sure it matches this inference pipeline (especially channels, normalization, and tensor shapes).

You said:
can you make offline mode as well?

ChatGPT said:

Yes — and you definitely should (it makes CNN testing way easier).

Here’s a version of your CNN inference script with both online + offline mode, matching your MLP workflow.
What it adds

    --mode online|offline

    --csv path_or_glob for offline testing

    per-class accuracy

    overall accuracy

    confusion matrix

    saves predictions + confusion matrix CSV

import os
import sys
import time
import glob
import datetime
import argparse
import numpy as np
import pandas as pd
import torch
from collections import deque
import socketio
import joblib
from sklearn.metrics import confusion_matrix

from semg_model import SEMGCNN, WINDOW_SIZE


# -----------------------------
# Globals (initialized in main)
# -----------------------------
model = None
le = None
buffer = None
WINDOW_SIZE_INT = int(WINDOW_SIZE)


def process_and_predict(data):
    global buffer, model, le
    try:
        raw = np.array(
            [
                data["ch0"]["a"],
                data["ch1"]["a"],
                data["ch2"]["a"],
                data["ch3"]["a"],
            ],
            dtype=np.float32,
        )

        buffer.append(raw)

        if len(buffer) == WINDOW_SIZE_INT:
            # buffer contains WINDOW_SIZE items of shape (4,)
            # stack -> (4, WINDOW_SIZE)
            window = np.stack(buffer, axis=1)

            # Normalize per channel (same as your online code)
            norm_signal = (window - window.mean(axis=1, keepdims=True)) / (
                window.std(axis=1, keepdims=True) + 1e-6
            )

            # CNN input shape: (batch, channels, time)
            tensor = torch.tensor(norm_signal, dtype=torch.float32).unsqueeze(0)  # (1, 4, WINDOW_SIZE)

            with torch.no_grad():
                out = model(tensor)
                pred = out.argmax(dim=1).item()

            # Non-overlap inference (same behavior as your MLP)
            buffer.clear()

            return le.inverse_transform([pred])[0]

    except Exception as e:
        print(f"Inference error: {e}")

    return None


# -----------------------------
# Socket.IO (online mode)
# -----------------------------
sio = socketio.Client()


@sio.event
def connect():
    print("Connected to Socket.IO server.")


@sio.on("adc_data")
def semg_data(data):
    prediction = process_and_predict(data)
    if prediction:
        print(f"Detection: {prediction}")


# -----------------------------
# Offline mode
# -----------------------------
def offline_predict(csv_path_pattern, skip_edge_rows=0):
    """
    Offline inference for one CSV or glob pattern.
    Assumes CSV columns: Ch0 Act, Ch1 Act, Ch2 Act, Ch3 Act
    Ground truth is inferred from filename (same logic as your MLP script).
    """
    global buffer, le

    if "*" in csv_path_pattern:
        csv_files = glob.glob(csv_path_pattern)
        if not csv_files:
            print("No CSV files found matching the pattern.")
            return
    else:
        csv_files = [csv_path_pattern]

    correct = 0
    total_predictions = 0

    TP = {cls: 0 for cls in le.classes_}
    Total_per_class = {cls: 0 for cls in le.classes_}

    all_actuals = []
    all_predictions = []
    prediction_records = []

    for csv_path in csv_files:
        buffer.clear()
        basename = os.path.basename(csv_path)

        # Infer ground truth from filename
        ground_truth = None
        lower_name = basename.lower()
        for cls in le.classes_:
            if str(cls).lower() in lower_name:
                ground_truth = cls
                break

        if ground_truth is None:
            print(f"Warning: Could not infer label from filename: {basename}. Skipping.")
            continue

        df = pd.read_csv(csv_path)

        required_cols = ["Ch0 Act", "Ch1 Act", "Ch2 Act", "Ch3 Act"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"Skipping {basename}: missing columns {missing}")
            continue

        # Optional: skip first/last rows to avoid transition noise
        start_idx = int(skip_edge_rows)
        end_idx = len(df) - int(skip_edge_rows)
        if end_idx <= start_idx:
            print(f"Skipping {basename}: too short after skip_edge_rows={skip_edge_rows}")
            continue

        for i in range(start_idx, end_idx):
            data = {
                "ch0": {"a": df.iloc[i]["Ch0 Act"]},
                "ch1": {"a": df.iloc[i]["Ch1 Act"]},
                "ch2": {"a": df.iloc[i]["Ch2 Act"]},
                "ch3": {"a": df.iloc[i]["Ch3 Act"]},
            }

            pred = process_and_predict(data)
            if pred:
                total_predictions += 1
                all_actuals.append(ground_truth)
                all_predictions.append(pred)

                prediction_records.append(
                    {
                        "file": basename,
                        "actual": ground_truth,
                        "predicted": pred,
                    }
                )

                Total_per_class[ground_truth] += 1
                if pred == ground_truth:
                    correct += 1
                    TP[ground_truth] += 1

        buffer.clear()

    # Metrics
    accuracy = correct / total_predictions if total_predictions > 0 else 0.0

    print("\nMetrics per class:")
    for cls in le.classes_:
        class_acc = TP[cls] / Total_per_class[cls] if Total_per_class[cls] > 0 else 0.0
        print(f"Class: {cls}")
        print(f"  Accuracy:  {class_acc:.4f}")

    print("\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}\n")

    # Save predictions CSV
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if prediction_records:
        results_filename = f"predictions_cnn_{timestamp}.csv"
        pd.DataFrame(prediction_records).to_csv(results_filename, index=False)
        print(f"Predictions saved to: {results_filename}")

    # Confusion matrix
    if all_actuals and all_predictions:
        cm = confusion_matrix(all_actuals, all_predictions, labels=le.classes_)
        cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)

        print("\nConfusion Matrix:")
        print("(Rows: Actual, Columns: Predicted)")
        print(cm_df)

        cm_filename = f"confusion_matrix_cnn_{timestamp}.csv"
        cm_df.to_csv(cm_filename)
        print(f"\nConfusion matrix saved to: {cm_filename}")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print(" ".join(sys.argv))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["online", "offline"],
        default="online",
        help="Detection mode",
    )
    parser.add_argument("--csv", type=str, help="CSV file or glob pattern for offline mode")
    parser.add_argument(
        "--skip_edge_rows",
        type=int,
        default=0,
        help="Skip this many rows at the start and end of each offline CSV (helps remove transitions)",
    )
    args = parser.parse_args()

    # Load label encoder and model
    le = joblib.load("label_encoder_cnn.pkl")
    n_classes = len(le.classes_)

    model = SEMGCNN(n_classes=n_classes)
    model.load_state_dict(torch.load("semg_cnn.pth", map_location="cpu"))
    model.eval()

    # Buffer for streaming inference
    buffer = deque(maxlen=WINDOW_SIZE_INT)

    if args.mode == "online":
        sio.connect("http://localhost:3000", transports=["websocket"], namespaces=["/"])
        try:
            print("waiting...")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            sio.disconnect()

    elif args.mode == "offline":
        if not args.csv:
            print("Please provide a CSV file or glob with --csv for offline mode.")
        else:
            offline_predict(args.csv, skip_edge_rows=args.skip_edge_rows)

How to run it
Online mode

python semg_infer_cnn.py --mode online

Offline on one file

python semg_infer_cnn.py --mode offline --csv "..\sutd_bmi_safety_data\unseen\grasp_1.csv"

Offline on all unseen files

python semg_infer_cnn.py --mode offline --csv "..\sutd_bmi_safety_data\unseen\*.csv"

Offline but skip start/end rows (recommended)

This helps reduce transition noise:

python semg_infer_cnn.py --mode offline --csv "..\sutd_bmi_safety_data\unseen\*.csv" --skip_edge_rows 50

One important check

Your SEMGCNN must use 4 input channels (not 2), since this script feeds 4 channels:

nn.Conv1d(in_channels=4, ...)

If you want, I can also give you a matching semg_train_cnn.py (offline training) that uses the same normalization + raw window format, so train/infer are fully consistent.


