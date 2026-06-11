import numpy as np
import pandas as pd
import joblib

def load_scaler(scaler_path="../models/scaler.pkl"):
    """Load the saved StandardScaler or MinMaxScaler."""
    return joblib.load(scaler_path)

def scale_input(data, scaler):
    """
    Scale sensor input before prediction.
    data: dict or list of sensor values
    scaler: loaded sklearn scaler
    """
    df = pd.DataFrame([data])
    scaled = scaler.transform(df)
    return scaled