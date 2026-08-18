import os
import json
import pandas as pd
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def main():
    print("=" * 65)
    print("      TRAINING TENSORFLOW DEEP LEARNING MODEL (INDIAN RAILWAYS)")
    print("=" * 65)

    base_dir = r"c:\Users\abhi\OneDrive\Desktop\SIH"
    data_dir = os.path.join(base_dir, "railways-master", "railways-master")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    stations_path = os.path.join(data_dir, "stations.json")
    trains_path = os.path.join(data_dir, "trains.json")
    schedules_path = os.path.join(data_dir, "schedules.json")

    print(f"Loading Indian Railways datasets...")
    with open(trains_path, 'r', encoding='utf-8') as f:
        trains_data = json.load(f)['features']

    with open(schedules_path, 'r', encoding='utf-8') as f:
        schedules_data = json.load(f)

    df_sched = pd.DataFrame(schedules_data)
    stops_per_train = df_sched.groupby('train_number').size().to_dict()

    train_records = []
    for feature in trains_data:
        props = feature.get('properties', {})
        t_num = props.get('number')
        dist = props.get('distance')
        dur_h = props.get('duration_h')
        dur_m = props.get('duration_m')

        if dur_h is not None and dur_m is not None:
            total_duration = float(dur_h) * 60.0 + float(dur_m)
        else:
            continue

        if dist is None or float(dist) <= 0 or total_duration <= 0:
            continue

        num_stops = stops_per_train.get(t_num, 0)
        t_type = props.get('type', 'Unknown')
        zone = props.get('zone', 'Unknown')

        dep_str = props.get('departure', '00:00:00')
        try:
            dep_hour = int(dep_str.split(':')[0])
        except Exception:
            dep_hour = 12

        train_records.append({
            'train_number': t_num,
            'distance': float(dist),
            'num_stops': num_stops,
            'train_type': str(t_type),
            'zone': str(zone),
            'departure_hour': dep_hour,
            'first_ac': 1 if props.get('first_ac') else 0,
            'second_ac': 1 if props.get('second_ac') else 0,
            'third_ac': 1 if props.get('third_ac') else 0,
            'sleeper': 1 if props.get('sleeper') else 0,
            'chair_car': 1 if props.get('chair_car') else 0,
            'target_duration_m': total_duration
        })

    df_trains = pd.DataFrame(train_records)
    print(f"Valid train samples for regression: {len(df_trains):,}")

    # One-Hot Encoding for train_type and zone
    df_trains = pd.get_dummies(df_trains, columns=['train_type', 'zone'], drop_first=True)

    feature_cols = [c for c in df_trains.columns if c not in ['train_number', 'target_duration_m']]

    X = df_trains[feature_cols].values.astype(np.float32)
    y = df_trains['target_duration_m'].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler and feature list
    joblib.dump({'scaler': scaler, 'feature_cols': feature_cols}, os.path.join(models_dir, "railways_tf_scaler.joblib"))

    print(f"Training Set size: {X_train.shape[0]:,}, Test Set size: {X_test.shape[0]:,}")

    # --- MODEL 3: TensorFlow Deep Neural Network Regressor ---
    print("\n--- Building & Training TensorFlow DNN Duration Regressor ---")

    model = models.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='linear')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
        loss=tf.keras.losses.Huber(delta=10.0),
        metrics=['mae', 'mse']
    )

    model.summary()

    early_stop = callbacks.EarlyStopping(monitor='val_mae', patience=8, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    model.fit(
        X_train_scaled, y_train,
        validation_split=0.15,
        epochs=35,
        batch_size=128,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Evaluate Regressor
    y_pred = model.predict(X_test_scaled).flatten()

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\nTensorFlow Duration Regressor Evaluation Results:")
    print(f"  R² Score                   : {r2:.4f}")
    print(f"  Mean Absolute Error (MAE)  : {mae:.2f} minutes ({mae/60:.2f} hours)")
    print(f"  Root Mean Squared Error    : {rmse:.2f} minutes ({rmse/60:.2f} hours)")

    save_path = os.path.join(models_dir, "railways_tf_regressor.keras")
    model.save(save_path)
    print(f"Saved TensorFlow Regressor to {save_path}")

    print("\n[SUCCESS] Indian Railways TensorFlow Deep Learning Model successfully trained & saved!")

if __name__ == "__main__":
    main()
