import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from semg_model import extract_features, WINDOW_SIZE
import joblib
import os
import argparse

def parse_feature_list(feature_str):
    feats = [x.strip().lower() for x in feature_str.split(",") if x.strip()]
    if not feats:
        raise ValueError("Feature list is empty.")
    return feats

if os.path.exists('semg_svm.pkl'):
    print("SVM model already exists. Exiting to avoid overwriting.")
    exit(0)

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="combined.csv")
parser.add_argument("--features", default="mav,rms,var,zc,ssc",
                    help="Comma-separated feature list")
parser.add_argument("--kernel", default="rbf", choices=["rbf", "linear", "poly"])
parser.add_argument("--C", type=float, default=10.0)
parser.add_argument("--gamma", default="scale")
parser.add_argument("--no-scaler", action="store_true", help="Disable StandardScaler")
args = parser.parse_args()

feature_names = parse_feature_list(args.features)
print("Using features:", feature_names)
print(f"SVM kernel={args.kernel}, C={args.C}, gamma={args.gamma}")

df = pd.read_csv(args.data)

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
            ]].values.T
        feats = extract_features(sample, feature_names=feature_names)
        label = chunk['Action'].iloc[0]
        X.append(feats)
        y.append(label)
    return np.stack(X), np.array(y)

X, y = create_samples(df, WINDOW_SIZE)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# StandardScaler — important for SVM
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Parse gamma
gamma = args.gamma
if gamma not in ("scale", "auto"):
    gamma = float(gamma)

svm = SVC(kernel=args.kernel, C=args.C, gamma=gamma, decision_function_shape='ovr')
print("Training SVM...")
svm.fit(X_train, y_train)

train_acc = svm.score(X_train, y_train)
val_acc = svm.score(X_test, y_test)
print(f"Train accuracy: {train_acc:.2%}")
print(f"Val accuracy:   {val_acc:.2%}")

# Save model, scaler, label encoder, and feature config
joblib.dump(svm, 'semg_svm.pkl')
joblib.dump(scaler, 'semg_svm_scaler.pkl')
joblib.dump(le, 'label_encoder_svm.pkl')
joblib.dump({"feature_names": feature_names}, "feature_config_svm.pkl")
print("SVM model saved.")

label_names = list(le.classes_)
print("Label mapping:")
for name, idx in zip(label_names, range(len(label_names))):
    print(f"  {name}: {idx}")
