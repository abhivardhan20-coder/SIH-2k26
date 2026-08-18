import os
import json
import joblib
import pandas as pd
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

def main():
    print("=" * 65)
    print("      TENSORFLOW DEEP LEARNING MODEL INFERENCE & TESTING DEMO")
    print("=" * 65)

    base_dir = r"c:\Users\abhi\OneDrive\Desktop\SIH"
    models_dir = os.path.join(base_dir, "models")

    # 1. METROPT TENSORFLOW INFERENCE
    clf_path = os.path.join(models_dir, "metro_tf_classifier.keras")
    auto_path = os.path.join(models_dir, "metro_tf_autoencoder.keras")
    scaler_path = os.path.join(models_dir, "metro_tf_scaler.joblib")

    if os.path.exists(clf_path) and os.path.exists(scaler_path):
        print("\n[1] METROPT TENSORFLOW DEEP LEARNING INFERENCE:")
        scaler_dict = joblib.load(scaler_path)
        scaler = scaler_dict['scaler']
        features = scaler_dict['features']
        sensor_cols = scaler_dict['sensor_cols']

        clf_model = tf.keras.models.load_model(clf_path)
        print("  Loaded TensorFlow Classifier (.keras)")

        if os.path.exists(auto_path):
            auto_model = tf.keras.models.load_model(auto_path)
            print("  Loaded TensorFlow Autoencoder (.keras)")

        # Normal Sample
        normal_reading = {
            'TP2': -0.012, 'TP3': 8.85, 'H1': 8.85, 'DV_pressure': -0.012,
            'Reservoirs': 8.85, 'Oil_temperature': 62.5, 'Motor_current': 6.8,
            'COMP': 1.0, 'DV_eletric': 1.0, 'Towers': 0.0, 'MPG': 1.0,
            'LPS': 0.0, 'Pressure_switch': 1.0, 'Oil_level': 1.0, 'Caudal_impulses': 1.0
        }
        for col in sensor_cols:
            normal_reading[f"{col}_roll_mean_6"] = normal_reading[col]
            normal_reading[f"{col}_roll_std_6"] = 0.01
            normal_reading[f"{col}_diff"] = 0.0

        df_norm = pd.DataFrame([normal_reading])[features]
        df_norm_scaled = scaler.transform(df_norm.values)

        prob_norm = clf_model.predict(df_norm_scaled, verbose=0)[0, 0]

        print("\n  Sample 1: Normal Compressor Operating State")
        print(f"    -> Failure / Air Leak Probability (TensorFlow): {prob_norm*100:.2f}%")
        print(f"    -> Anomaly Status                             : {'[AIR LEAK RISK]' if prob_norm >= 0.5 else '[NORMAL HEALTH]'}")

        # Degraded Sample
        stress_reading = normal_reading.copy()
        stress_reading['Reservoirs'] = 5.2
        stress_reading['Oil_temperature'] = 82.0
        stress_reading['Motor_current'] = 9.4
        for col in sensor_cols:
            stress_reading[f"{col}_roll_mean_6"] = stress_reading[col]
            stress_reading[f"{col}_roll_std_6"] = 0.8
            stress_reading[f"{col}_diff"] = -0.5

        df_stress = pd.DataFrame([stress_reading])[features]
        df_stress_scaled = scaler.transform(df_stress.values)

        prob_stress = clf_model.predict(df_stress_scaled, verbose=0)[0, 0]

        print("\n  Sample 2: Degraded / Air Leak Stress Sensor Reading")
        print(f"    -> Failure / Air Leak Probability (TensorFlow): {prob_stress*100:.2f}%")
        print(f"    -> Anomaly Status                             : {'[AIR LEAK RISK]' if prob_stress >= 0.5 else '[NORMAL HEALTH]'}")
    else:
        print("\n[1] TensorFlow MetroPT model not found. Run train_metro_tf.py first.")

    # 2. RAILWAYS TENSORFLOW INFERENCE
    rail_path = os.path.join(models_dir, "railways_tf_regressor.keras")
    rail_scaler_path = os.path.join(models_dir, "railways_tf_scaler.joblib")

    if os.path.exists(rail_path) and os.path.exists(rail_scaler_path):
        print("\n" + "=" * 65)
        print("[2] INDIAN RAILWAYS TENSORFLOW DURATION REGRESSOR INFERENCE:")
        rail_scaler_dict = joblib.load(rail_scaler_path)
        rail_scaler = rail_scaler_dict['scaler']
        feature_cols = rail_scaler_dict['feature_cols']

        rail_model = tf.keras.models.load_model(rail_path)
        print("  Loaded TensorFlow Duration Regressor (.keras)")

        sample_train = {col: 0.0 for col in feature_cols}
        sample_train['distance'] = 1200.0
        sample_train['num_stops'] = 18.0
        sample_train['departure_hour'] = 16.0
        sample_train['first_ac'] = 1.0
        sample_train['second_ac'] = 1.0
        sample_train['third_ac'] = 1.0
        sample_train['sleeper'] = 1.0

        for col in feature_cols:
            if 'train_type_SF' in col:
                sample_train[col] = 1.0
            elif 'zone_NR' in col:
                sample_train[col] = 1.0

        df_q = pd.DataFrame([sample_train])[feature_cols]
        df_q_scaled = rail_scaler.transform(df_q.values)

        pred_mins = rail_model.predict(df_q_scaled, verbose=0)[0, 0]

        print("\n  Sample Query: Superfast Train (1,200 km, 18 stops, NR Zone)")
        print(f"    -> TensorFlow Predicted Duration: {pred_mins:.1f} mins ({pred_mins/60:.2f} hours)")
    else:
        print("\n[2] TensorFlow Railways model not found. Run train_railways_tf.py first.")

    print("\n" + "=" * 65)

if __name__ == "__main__":
    main()
