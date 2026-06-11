import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from .preprocessing import scale_input, load_scaler

# Load autoencoder model
def load_autoencoder(model_path="../models/autoencoder_model.keras"):
    """Load the trained Autoencoder anomaly detection model."""
    model = load_model(model_path)
    return model

# Load threshold for anomaly detection
def load_threshold(path="../models/threshold.txt"):
    with open(path, "r") as f:
        return float(f.read().strip())

# Calculate reconstruction error
def reconstruction_error(autoencoder, scaled_data):
    reconstructed = autoencoder.predict(scaled_data, verbose=0)
    error = np.mean(np.square(scaled_data - reconstructed))
    return float(error)

# Detect anomaly based on threshold
def detect_anomaly(raw_data, scaler, autoencoder, threshold):
    """
    raw_data: list or dict of sensor values
    scaler: loaded scaler from preprocessing
    autoencoder: loaded AE model
    threshold: numeric threshold for anomaly detection
    """
    scaled = scale_input(raw_data, scaler)
    error = reconstruction_error(autoencoder, scaled)

    is_anomaly = error > threshold
    health_score = max(0, 1 - (error / (threshold * 3)))  # simple health formula

    return {
        "reconstruction_error": error,
        "is_anomaly": bool(is_anomaly),
        "health_score": round(health_score, 4)
    }