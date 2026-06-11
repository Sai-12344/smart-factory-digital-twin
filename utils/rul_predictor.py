import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from .preprocessing import load_scaler, scale_input

# Load LSTM model
def load_lstm_model(model_path="../models/lstm_rul_model.keras"):
    """Load the trained LSTM RUL model."""
    model = load_model(model_path)
    return model

# Prepare sequence for LSTM (sliding window)
def create_sequence(data, seq_length=30):
    """
    Prepares time-series data into LSTM-ready format.
    data: list or array of sensor snapshots
    """
    sequence = np.array(data[-seq_length:])
    sequence = sequence.reshape(1, seq_length, sequence.shape[1])
    return sequence

# RUL prediction function
def predict_rul(sensor_sequence, scaler, model):
    """
    Predict Remaining Useful Life using trained LSTM model.
    sensor_sequence: list of raw sensor rows (each row = all sensor values)
    scaler: loaded scaler
    model: LSTM model
    """
    # Scale entire sequence
    scaled_seq = [scale_input(row, scaler)[0] for row in sensor_sequence]

    # Convert to LSTM input format
    seq = create_sequence(scaled_seq)

    # Model prediction
    rul = model.predict(seq)[0][0]
    return float(rul)