import os
import glob
import argparse
import numpy as np
import pandas as pd
import sys
import joblib
from semg_model import extract_features, WINDOW_SIZE

# Load SVM model, scaler, label encoder, feature config
svm = joblib.load("semg_svm.pkl")
scaler = joblib.load("semg_svm_scaler.pkl")
le = joblib.load("label_encoder_svm.pkl")
feature_config = joblib.load("feature_config_svm.pkl")
feature_names = feature_config["feature_names"]

print("Classes:", list(le.classes_))
print("Using features:", feature_names)


def offline_predict(csv_path):
    if "*" in csv_path:
        csv_files = sorted(glob.glob(csv_path, recursive=True))
        if not csv_files:
            print("No CSV files found matching the pattern.")
            return
    else:
        csv_files = [csv_path]

    TP = {cls: 0 for cls in le.classes_}
    FP = {cls: 0 for cls in le.classes_}
    Total_per_class = {cls: 0 for cls in le.classes_}
    correct = 0
    total_predictions = 0

    for one_csv in csv_files:
        basename = os.path.basename(one_csv)
        ground_truth = None
        parts = basename.split("_")
        if len(parts) >= 2:
            gesture = parts[1]
            for cls in le.classes_:
                if cls.startswith(gesture):
                    ground_truth = cls
                    break

        if ground_truth is None:
            continue

        df = pd.read_csv(one_csv)
        for i in range(0, len(df), WINDOW_SIZE):
            chunk = df.iloc[i:i + WINDOW_SIZE]
            if len(chunk) < WINDOW_SIZE:
                continue
            sample = chunk[['Ch0 Act', 'Ch1 Act', 'Ch2 Act', 'Ch3 Act']].values.T
            feats = extract_features(sample, feature_names=feature_names)
            feats_scaled = scaler.transform(feats.reshape(1, -1))
            pred = le.inverse_transform(svm.predict(feats_scaled))[0]

            total_predictions += 1
            if pred == ground_truth:
                correct += 1
                TP[ground_truth] += 1
            else:
                FP[pred] += 1
            Total_per_class[ground_truth] += 1

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
    parser.add_argument("--mode", choices=["offline"], required=True)
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    if args.mode == "offline":
        offline_predict(args.csv)
