import os
import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_fscore_support
import joblib

def main():
    print("=" * 65)
    print("      TRAINING DEEP NEURAL NETWORK MODELS (METROPT-3)")
    print("=" * 65)

    base_dir = r"c:\Users\abhi\OneDrive\Desktop\SIH"
    csv_path = os.path.join(base_dir, "metropt_3_dataset", "MetroPT3(AirCompressor).csv")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    print(f"Reading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Subsample every 5th row
    df = df.iloc[::5].reset_index(drop=True)

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n--- Training Deep Neural Network (MLP: 128 -> 64 -> 32) ---")
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=512,
        learning_rate_init=0.001,
        max_iter=30,
        random_state=42,
        early_stopping=True,
        verbose=True
    )

    mlp.fit(X_train_scaled, y_train)

    y_prob = mlp.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

    print(f"\nDeep Neural Network Classifier Evaluation Results:")
    print(f"  ROC-AUC Score : {auc:.4f}")
    print(f"  Precision     : {prec:.4f}")
    print(f"  Recall        : {rec:.4f}")
    print(f"  F1 Score      : {f1:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    model_path = os.path.join(models_dir, "metro_neural_classifier.joblib")
    joblib.dump({'model': mlp, 'scaler': scaler, 'features': features, 'auc': auc, 'f1': f1}, model_path)
    print(f"Saved Deep Neural Network Classifier to {model_path}")

if __name__ == "__main__":
    main()
