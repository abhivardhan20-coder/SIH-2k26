import os
import pandas as pd
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_fscore_support
import joblib

def main():
    print("=" * 65)
    print("      TRAINING TENSORFLOW DEEP LEARNING MODELS (METROPT-3)")
    print("=" * 65)

    base_dir = r"c:\Users\abhi\OneDrive\Desktop\SIH"
    csv_path = os.path.join(base_dir, "metropt_3_dataset", "MetroPT3(AirCompressor).csv")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    print(f"Reading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} raw rows.")

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Subsample every 5th row for speed
    df = df.iloc[::5].reset_index(drop=True)

    # Define failure windows from maintenance report
    failure_windows = [
        (pd.to_datetime('2020-04-18 00:00:00'), pd.to_datetime('2020-04-18 23:59:59')),
        (pd.to_datetime('2020-05-29 23:30:00'), pd.to_datetime('2020-05-30 06:00:00')),
        (pd.to_datetime('2020-06-05 10:00:00'), pd.to_datetime('2020-06-07 14:30:00')),
        (pd.to_datetime('2020-07-15 14:30:00'), pd.to_datetime('2020-07-15 19:00:00')),
    ]

    df['is_failure_stress'] = 0
    for start_t, end_t in failure_windows:
        pre_t = start_t - pd.Timedelta(hours=2)
        mask = (df['timestamp'] >= pre_t) & (df['timestamp'] <= end_t)
        df.loc[mask, 'is_failure_stress'] = 1

    # Feature Engineering
    sensor_cols = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Oil_temperature', 'Motor_current']
    digital_cols = ['COMP', 'DV_eletric', 'Towers', 'MPG', 'LPS', 'Pressure_switch', 'Oil_level', 'Caudal_impulses']

    features = sensor_cols + digital_cols
    for col in sensor_cols:
        mean_col = f"{col}_roll_mean_6"
        std_col = f"{col}_roll_std_6"
        diff_col = f"{col}_diff"
        df[mean_col] = df[col].rolling(window=6, min_periods=1).mean()
        df[std_col] = df[col].rolling(window=6, min_periods=1).std().fillna(0)
        df[diff_col] = df[col].diff().fillna(0)
        features.extend([mean_col, std_col, diff_col])

    df = df.dropna().reset_index(drop=True)

    X = df[features].values
    y = df['is_failure_stress'].values

    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler for inference
    scaler_path = os.path.join(models_dir, "metro_tf_scaler.joblib")
    joblib.dump({'scaler': scaler, 'features': features, 'sensor_cols': sensor_cols}, scaler_path)

    # Calculate class weights to handle imbalance (~2.16% failure class)
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    total = len(y_train)
    class_weight = {
        0: (1 / neg_count) * (total / 2.0),
        1: (1 / pos_count) * (total / 2.0)
    }

    print(f"Dataset prepared: {X_train.shape[0]:,} Train samples, {X_test.shape[0]:,} Test samples.")

    # --- MODEL 1: TensorFlow Deep Neural Network Failure Classifier ---
    print("\n--- [1] Building & Training TensorFlow Predictive Maintenance Classifier ---")

    classifier_model = models.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    classifier_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Recall(name='recall')]
    )

    classifier_model.summary()

    early_stop = callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=5, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

    history = classifier_model.fit(
        X_train_scaled, y_train,
        validation_split=0.15,
        epochs=15,
        batch_size=512,
        class_weight=class_weight,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Evaluate Classifier on Test Set
    y_prob = classifier_model.predict(X_test_scaled, batch_size=1024).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

    print(f"\nTensorFlow Classifier Evaluation Results:")
    print(f"  ROC-AUC Score : {auc:.4f}")
    print(f"  Precision     : {prec:.4f}")
    print(f"  Recall        : {rec:.4f}")
    print(f"  F1 Score      : {f1:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    clf_save_path = os.path.join(models_dir, "metro_tf_classifier.keras")
    classifier_model.save(clf_save_path)
    print(f"Saved TensorFlow Classifier to {clf_save_path}")

    # --- MODEL 2: TensorFlow Deep Autoencoder Anomaly Detector ---
    print("\n--- [2] Building & Training TensorFlow Deep Autoencoder Anomaly Detector ---")

    # Filter healthy sensor observations for autoencoder training
    sensor_indices = [features.index(c) for c in sensor_cols]
    X_train_sensors = X_train_scaled[:, sensor_indices]
    X_test_sensors = X_test_scaled[:, sensor_indices]

    healthy_mask = (y_train == 0)
    X_train_healthy = X_train_sensors[healthy_mask]

    input_dim = len(sensor_cols)

    # Encoder
    inputs = layers.Input(shape=(input_dim,))
    encoder = layers.Dense(32, activation='relu')(inputs)
    encoder = layers.Dense(16, activation='relu')(encoder)
    bottleneck = layers.Dense(8, activation='relu')(encoder)

    # Decoder
    decoder = layers.Dense(16, activation='relu')(bottleneck)
    decoder = layers.Dense(32, activation='relu')(decoder)
    outputs = layers.Dense(input_dim, activation='linear')(decoder)

    autoencoder = models.Model(inputs=inputs, outputs=outputs)
    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(
        X_train_healthy, X_train_healthy,
        epochs=10,
        batch_size=512,
        validation_split=0.1,
        verbose=1
    )

    # Evaluate reconstruction error on test set
    reconstructions = autoencoder.predict(X_test_sensors, batch_size=1024)
    mse_errors = np.mean(np.square(X_test_sensors - reconstructions), axis=1)

    threshold = np.percentile(mse_errors[y_test == 0], 95) # 95th percentile threshold
    anomalies_detected = (mse_errors > threshold).sum()

    print(f"\nAutoencoder Evaluation Results:")
    print(f"  95th Percentile Anomaly Threshold (MSE): {threshold:.4f}")
    print(f"  Test Set Anomalies Flagged            : {anomalies_detected:,} out of {len(y_test):,}")

    autoencoder_save_path = os.path.join(models_dir, "metro_tf_autoencoder.keras")
    autoencoder.save(autoencoder_save_path)
    print(f"Saved TensorFlow Autoencoder to {autoencoder_save_path}")

    print("\n[SUCCESS] MetroPT-3 TensorFlow Deep Learning Models successfully trained & saved!")

if __name__ == "__main__":
    main()
