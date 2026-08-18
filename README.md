# Smart Transit & Maintenance AI Systems

This repository contains machine learning and deep learning pipelines designed for two transportation and infrastructure domains:
1. **MetroPT-3 Predictive Maintenance & Anomaly Detection** (Air Compressor Failures in Metro Trains)
2. **Indian Railways Journey Duration Prediction** (Trip Travel Time Regression)

---

## 📁 Generated Models & Documentation

### 1. MetroPT-3 Predictive Maintenance & Anomaly Detection

#### 🔹 `models/metro_neural_classifier.joblib`
* **Framework**: Scikit-Learn (`MLPClassifier`)
* **Architecture**: 3-layer Deep Multi-Layer Perceptron (`128 -> 64 -> 32` hidden units with ReLU activation and Adam optimizer).
* **Dataset Used**: `metropt_3_dataset/MetroPT3(AirCompressor).csv` (High-frequency time-series telemetry sensor logs from Metro train Air Processing Units).
* **Target Output**: Binary Classification (`0` = Normal Health, `1` = Air Leak / Failure Stress Risk).
* **Feature Inputs**: 29 engineered features including analog pressure signals (`TP2`, `TP3`, `H1`, `DV_pressure`, `Reservoirs`), oil temperature, motor current, digital valve states, along with 6-period rolling means, rolling standard deviations, and step differences.
* **What it can be used for**: Real-time early warning system on metro trains to alert maintenance teams about impending air compressor failure hours before track breakdown occurs.

#### 🔹 `models/metro_tf_classifier.keras`
* **Framework**: TensorFlow / Keras 3 (`tf.keras.models.Sequential`)
* **Architecture**: Deep Neural Network (`128 -> 64 -> 32`) with `BatchNormalization`, `Dropout (0.3)`, binary cross-entropy loss, and balanced class weights to handle severe class imbalance (~2.16% failure window class).
* **Dataset Used**: `metropt_3_dataset/MetroPT3(AirCompressor).csv`.
* **Target Output**: Probability score ($0.0 - 1.0$) indicating air compressor failure / air leak risk.
* **What it can be used for**: Production cloud/edge inference service for real-time risk scoring and automated maintenance ticket generation.

#### 🔹 `models/metro_tf_autoencoder.keras`
* **Framework**: TensorFlow / Keras 3 (`tf.keras.Model`)
* **Architecture**: Deep Autoencoder Anomaly Detector (`Input(7) -> 32 -> 16 -> 8 -> 16 -> 32 -> Output(7)`).
* **Dataset Used**: `metropt_3_dataset/MetroPT3(AirCompressor).csv` (Trained exclusively on healthy baseline operating sensor observations).
* **Target Output**: Reconstruction Error (Mean Squared Error) against a 95th-percentile anomaly threshold (`0.0019`).
* **What it can be used for**: Unsupervised anomaly detection to discover novel, previously unseen mechanical faults (e.g., valve degradation, oil contamination, pump friction) that do not match historical failure logs.

#### 🔹 `models/metro_tf_scaler.joblib`
* **Framework**: Scikit-Learn (`StandardScaler`)
* **Dataset Used**: `metropt_3_dataset/MetroPT3(AirCompressor).csv`.
* **What it can be used for**: Normalizing raw sensor inputs (zero mean, unit variance) prior to feeding into `metro_tf_classifier.keras` or `metro_tf_autoencoder.keras`.

---

### 2. Indian Railways Journey Duration Prediction

#### 🔹 `models/railways_neural_regressor.joblib`
* **Framework**: Scikit-Learn (`MLPRegressor`)
* **Architecture**: 3-layer Deep Multi-Layer Perceptron Regressor (`256 -> 128 -> 64` hidden units with ReLU activation).
* **Dataset Used**: Indian Railways Datasets (`railways-master/railways-master/trains.json` and `schedules.json`).
* **Target Output**: Continuous predicted total journey time in **minutes**.
* **Feature Inputs**: Route distance (km), total station stops, departure hour, coach availability flags (1AC, 2AC, 3AC, Sleeper, Chair Car), and one-hot encoded train categories (Superfast, Express, etc.) and railway operating zones (NR, WR, SR, etc.).
* **What it can be used for**: Rapid travel duration estimation for passenger ticketing apps and railway scheduling tools.

#### 🔹 `models/railways_tf_regressor.keras`
* **Framework**: TensorFlow / Keras 3 (`tf.keras.models.Sequential`)
* **Architecture**: Deep Neural Network Regressor (`Input -> 256 -> 128 -> 64 -> 32 -> 1`) utilizing Huber loss ($\delta=10.0$), `BatchNormalization`, `Dropout (0.2)`, and `ReduceLROnPlateau`.
* **Dataset Used**: Indian Railways Datasets (`railways-master/railways-master/trains.json`, `schedules.json`, `stations.json`).
* **Performance**: **$R^2 = 0.9837$** (98.37% variance explained), MAE = `61.64 minutes` (~1.03 hours across long-distance routes).
* **Target Output**: Continuous predicted trip duration in **minutes** (and hours).
* **What it can be used for**: High-precision Estimated Time of Arrival (ETA) engines for passenger apps, timetable optimization, and logistics/fleet management.

#### 🔹 `models/railways_tf_scaler.joblib`
* **Framework**: Scikit-Learn (`StandardScaler`)
* **Dataset Used**: Indian Railways tabular features.
* **What it can be used for**: Normalizing route parameters before executing inference with `railways_tf_regressor.keras`.

---

## 🛠️ How to Run Scripts

### 1. Train Models
```bash
# Train Scikit-Learn MetroPT Classifier
python train_metro_neural.py

# Train TensorFlow MetroPT Classifier & Autoencoder
python train_metro_tf.py

# Train Scikit-Learn Railways Duration Regressor
python train_railways_neural.py

# Train TensorFlow Railways Duration Regressor
python train_railways_tf.py
```

### 2. Run Inference Demo
```bash
python demo_tf_inference.py
```

---

## 📂 Directory Structure

```text
SIH/
├── models/                         # Serialized Model Artifacts
│   ├── metro_neural_classifier.joblib
│   ├── metro_tf_classifier.keras
│   ├── metro_tf_autoencoder.keras
│   ├── metro_tf_scaler.joblib
│   ├── railways_neural_regressor.joblib
│   ├── railways_tf_regressor.keras
│   └── railways_tf_scaler.joblib
├── metropt_3_dataset/              # MetroPT Sensor Dataset & PDF Docs
│   ├── MetroPT3(AirCompressor).csv
│   └── Data Description_Metro.pdf
├── railways-master/                # Indian Railways JSON Datasets
│   └── railways-master/
│       ├── trains.json
│       ├── schedules.json
│       └── stations.json
├── train_metro_neural.py           # Scikit-Learn MetroPT Training Script
├── train_metro_tf.py               # TensorFlow MetroPT Training Script
├── train_railways_neural.py        # Scikit-Learn Railways Training Script
├── train_railways_tf.py            # TensorFlow Railways Training Script
├── demo_tf_inference.py            # End-to-End TensorFlow Inference Test
└── README.md                       # Comprehensive Project Documentation
```
