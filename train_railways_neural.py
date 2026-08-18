import os
import json
import pandas as pd
import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def main():
    print("=" * 65)
    print("      TRAINING DEEP NEURAL NETWORK REGRESSOR (INDIAN RAILWAYS)")
    print("=" * 65)

    base_dir = r"c:\Users\abhi\OneDrive\Desktop\SIH"
    data_dir = os.path.join(base_dir, "railways-master", "railways-master")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

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

    df_trains = pd.get_dummies(df_trains, columns=['train_type', 'zone'], drop_first=True)
    feature_cols = [c for c in df_trains.columns if c not in ['train_number', 'target_duration_m']]

    X = df_trains[feature_cols].values.astype(np.float32)
    y = df_trains['target_duration_m'].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n--- Training Deep Neural Network Regressor (MLP: 256 -> 128 -> 64) ---")
    mlp_reg = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=128,
        learning_rate_init=0.003,
        max_iter=60,
        random_state=42,
        early_stopping=True,
        verbose=True
    )

    mlp_reg.fit(X_train_scaled, y_train)

    y_pred = mlp_reg.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\nDeep Neural Network Duration Regressor Evaluation Results:")
    print(f"  R² Score                   : {r2:.4f}")
    print(f"  Mean Absolute Error (MAE)  : {mae:.2f} minutes ({mae/60:.2f} hours)")
    print(f"  Root Mean Squared Error    : {rmse:.2f} minutes ({rmse/60:.2f} hours)")

    save_path = os.path.join(models_dir, "railways_neural_regressor.joblib")
    joblib.dump({'model': mlp_reg, 'scaler': scaler, 'feature_cols': feature_cols, 'r2': r2, 'mae': mae}, save_path)
    print(f"Saved Deep Neural Network Regressor to {save_path}")

if __name__ == "__main__":
    main()
