
# Real-Time sEMG Classification System

This project implements a complete machine learning pipeline for the classification of surface electromyography (sEMG) signals in real-time using a neural network model. The offline pipeline now uses four activation channels and hand-crafted features, while the real-time WebSocket inference flow remains in place for streaming use.

---

## 📁 Dataset

The dataset used is in `combined.csv` and contains the following columns:

- `Timestamp`: The timestamp of the sample (ISO string)
- `Ch0 Act`: Activation level of sEMG Channel 0
- `Ch0 Env`: Envelope of sEMG Channel 0
- `Ch1 Act`: Activation level of sEMG Channel 1
- `Ch1 Env`: Envelope of sEMG Channel 1
- `Action`: The class label representing a specific physical action

Each row represents one timestamped sample in the recording. The offline pipeline groups 100 consecutive rows into one window for feature extraction and classification.

---

## 🧠 Machine Learning Pipeline

### 1. Preprocessing
- **Feature Selection**: The offline pipeline uses the four activation channels `Ch0 Act`, `Ch1 Act`, `Ch2 Act`, and `Ch3 Act`.
- **Windowing**: Each sample is built from 100 rows, which matches `WINDOW_SIZE` in the current code.
- **Feature Extraction**: The default feature set is `mav`, `rms`, `var`, `zc`, `ssc`.
- **Scaling**: `StandardScaler` is used for the SVM pipeline.
- **Label Encoding**: Action labels are encoded into integers using `LabelEncoder`.

### 2. Sequence Generation
The old sequence-based CNN path is no longer used for the offline models. Instead, the data is grouped into fixed 100-row windows and converted into a single feature vector per window:

```python
def create_samples(df, window_size):
  # take Ch0/Ch1/Ch2/Ch3 Act columns
  # extract mav, rms, var, zc, ssc from each 100-row window
  # return feature vectors and labels
```

This gives one feature vector per 100-row window rather than a sliding temporal sequence.

### 3. Model Architecture

The offline classifier is a feed-forward MLP with the following structure:

- `Linear(input_dim, 128)`
- `ReLU()`
- `Dropout(0.1)`
- `Linear(128, 64)`
- `ReLU()`
- `Dropout(0.1)`
- `Linear(64, n_classes)`

Loss: Cross-entropy  
Optimizer: Adam

The SVM alternative uses `sklearn.svm.SVC` with scaled feature vectors.

The MLP trainer uses an 80/20 train-test split, early stopping, and saves the best checkpoint.

---

## 💾 Saving Model Artifacts

After offline training:
- MLP model is saved as `semg_mlp.pth`
- MLP label encoder is saved as `label_encoder_mlp.pkl`
- MLP feature config is saved as `feature_config.pkl`
- Calibrated MLP model is saved as `semg_mlp_calibrated.pth` when calibration is run
- SVM model is saved as `semg_svm.pkl`
- SVM scaler is saved as `semg_svm_scaler.pkl`
- SVM label encoder is saved as `label_encoder_svm.pkl`
- SVM feature config is saved as `feature_config_svm.pkl`

---

## 🔌 Real-Time Inference (WebSocket)

A FastAPI server listens for real-time data streamed from a WebSocket (`localhost:3000/ws`). Incoming JSON messages are expected in the following format:

```json
{
  "ch0": { "a": float, "e": float },
  "ch1": { "a": float, "e": float }
}
```

These are appended to a buffer (`deque` of size 50). Once the buffer is full, the model performs inference on the sliding window and returns a classification label.

### Server Launch

```bash
uvicorn semg_websocket_server:app --reload --host 0.0.0.0 --port 8000
```

### Output

The server returns predicted class labels through the `/ws` WebSocket endpoint.

---

## 📦 Deployment Stack

- **Language**: Python 3.8+
- **Libraries**:
  - TensorFlow (for model training and inference)
  - FastAPI (for WebSocket server)
  - websockets (for internal proxy to data source)
  - scikit-learn (for preprocessing)

---

## 🧪 Testing

You can simulate real-time data by emitting JSON packets with the appropriate structure using Node.js or another WebSocket client.

---

## 📄 For Thesis Reference

This project demonstrates:
- Time-series classification using deep learning
- Preprocessing and modeling of biosignals
- Real-time streaming classification architecture
- Integration of machine learning with WebSocket I/O
- End-to-end deployment from CSV to inference

---
