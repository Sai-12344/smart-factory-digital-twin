import numpy as np
import cv2
import tensorflow as tf


class DroneInspector:
    """
    Lightweight AI module for drone inspection:
    - Thermal hotspot detection
    - CNN-based anomaly classification (optional)
    - Risk scoring for each machine
    """

    def __init__(self, cnn_model_path=None):
        self.cnn_model = None

        if cnn_model_path:
            try:
                self.cnn_model = tf.keras.models.load_model(cnn_model_path)
                print("Drone CNN model loaded.")
            except:
                print("Warning: CNN model not found. Using thermal detection only.")

    # ------------------------------------------------------
    # THERMAL HOTSPOT DETECTION
    # ------------------------------------------------------
    def detect_hotspots(self, heatmap):
        """
        heatmap: 2D array (temperature map)
        returns: list of hotspot coords + intensity
        """

        hotspots = []
        threshold = np.percentile(heatmap, 90)  # top 10% = hot

        for i in range(heatmap.shape[0]):
            for j in range(heatmap.shape[1]):
                if heatmap[i, j] >= threshold:
                    hotspots.append({
                        "x": i,
                        "y": j,
                        "intensity": float(heatmap[i, j])
                    })

        return hotspots

    # ------------------------------------------------------
    # CNN ANOMALY CLASSIFICATION (optional)
    # ------------------------------------------------------
    def classify_frame(self, frame):
        """
        frame: image array
        returns: anomaly label + confidence
        """
        if self.cnn_model is None:
            return "unknown", 0.0

        frame = cv2.resize(frame, (64, 64)) / 255.0
        frame = np.expand_dims(frame, axis=0)

        pred = self.cnn_model.predict(frame)[0]
        label = np.argmax(pred)
        confidence = pred[label]

        return label, float(confidence)

    # ------------------------------------------------------
    # COMBINED RISK SCORE
    # ------------------------------------------------------
    def compute_risk_score(self, failure_prob, hotspots):
        """
        Combine Digital Twin physics + drone hotspot info.
        """

        # Base risk from digital twin:
        risk = failure_prob * 0.6

        # Add hotspot effect
        if len(hotspots) > 0:
            heat_intensity = np.mean([h["intensity"] for h in hotspots])
            risk += (heat_intensity / 100) * 0.4

        return min(1.0, float(risk))