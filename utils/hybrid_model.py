from .rul_predictor import predict_rul
from .anomaly_detector import detect_anomaly

def hybrid_rul_prediction(sensor_sequence, raw_latest_row, scaler, lstm_model, autoencoder, threshold):
    """
    Combines RUL prediction + anomaly detection for more reliable maintenance decisions.

    sensor_sequence  : List of past sensor rows (for LSTM)
    raw_latest_row   : Latest sensor reading (for AE)
    scaler           : Fitted scaler
    lstm_model       : Loaded LSTM RUL model
    autoencoder      : Loaded AE model
    threshold        : Threshold for anomaly score
    """

    # --- Step 1: Base RUL prediction from LSTM ---
    base_rul = predict_rul(sensor_sequence, scaler, lstm_model)

    # --- Step 2: Anomaly detection from Autoencoder ---
    anomaly_info = detect_anomaly(raw_latest_row, scaler, autoencoder, threshold)
    health_score = anomaly_info["health_score"]
    is_anomaly = anomaly_info["is_anomaly"]

    # --- Step 3: Hybrid RUL adjustment ---
    # If anomaly is detected, reduce RUL proportionally
    if is_anomaly:
        adjusted_rul = base_rul * (0.4 + health_score * 0.6)
    else:
        adjusted_rul = base_rul * (0.8 + health_score * 0.2)

    # Ensure RUL doesn't go negative
    adjusted_rul = max(0, adjusted_rul)

    return {
        "base_rul": float(base_rul),
        "adjusted_rul": float(adjusted_rul),
        "is_anomaly": is_anomaly,
        "reconstruction_error": anomaly_info["reconstruction_error"],
        "health_score": anomaly_info["health_score"]
    }