import os
import numpy as np
import torch
from collections import deque
import socketio
import time
from semg_model import extract_features, SEMGMLP, WINDOW_SIZE, get_feature_dim
import joblib
import argparse
import pandas as pd
import sys

# Load label encoder, feature config, and model
le = joblib.load("label_encoder_mlp.pkl")
feature_config = joblib.load("feature_config.pkl")
feature_names = feature_config["feature_names"]

print("Classes:", list(le.classes_))
print("Using features:", feature_names)

input_dim = get_feature_dim(n_channels=4, feature_names=feature_names)
n_classes = len(le.classes_)
model = SEMGMLP(n_classes, input_dim)

# Use calibrated model if available, otherwise use base model
if os.path.exists("semg_mlp_calibrated.pth"):
    model.load_state_dict(torch.load("semg_mlp_calibrated.pth", map_location="cpu"))
    print("Loaded calibrated model")
else:
    model.load_state_dict(torch.load("semg_mlp.pth", map_location="cpu"))
    print("Loaded base model (no calibration)")
model.eval()

# Initialize buffer
buffer = deque(maxlen=WINDOW_SIZE)


def process_and_predict(data):
    try:
        raw = np.array(
            [
                data["ch0"]["a"],
                data["ch1"]["a"],
                data["ch2"]["a"],
                data["ch3"]["a"],
            ]
        )
        buffer.append(raw)
        if len(buffer) == WINDOW_SIZE:
            window = np.stack(buffer, axis=1)
            feats = extract_features(window, feature_names=feature_names)
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
        csv_files = glob.glob(csv_path, recursive=True)
        if not csv_files:
            print("No CSV files found matching the pattern.")
            return
    else:
        csv_files = [csv_path]

    correct = 0

    TP = {cls: 0 for cls in le.classes_}
    FP = {cls: 0 for cls in le.classes_}
    Total_per_class = {cls: 0 for cls in le.classes_}
    total_predictions = 0

    for csv_path in csv_files:
        buffer.clear()
        basename = os.path.basename(csv_path)
        ground_truth = None
        parts = basename.split("_")
        # Allow partial matching of gesture name for flexibility in naming conventions of filename
        if len(parts) >= 2:
            gesture = parts[1]
            for cls in le.classes_:
                if cls.startswith(gesture):
                    ground_truth = cls
                    break

        df = pd.read_csv(csv_path)
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
                if pred == ground_truth:
                    correct += 1
                    TP[ground_truth] += 1
                else:
                    FP[pred] += 1

                Total_per_class[ground_truth] += 1
        buffer.clear()

    accuracy = correct / total_predictions if total_predictions > 0 else 0.0

    print("\nMetrics per class:")
    for cls in le.classes_:
        class_accuracy = TP[cls] / Total_per_class[cls] if Total_per_class[cls] > 0 else 0.0
        print(f"Class: {cls}")
        print(f"  Accuracy:  {class_accuracy:.4f}")

    print("\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}\n")


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
