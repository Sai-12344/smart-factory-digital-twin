import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class ThermalCNN:
    """
    CNN for drone thermal anomaly classification.
    Labels can include:
    - 0 = normal
    - 1 = hotspot
    - 2 = crack-like anomaly
    - 3 = smoke/irregular pattern
    """

    def __init__(self, input_shape=(64, 64, 1), num_classes=4):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None

    # --------------------------------------------------------
    # BUILD CNN MODEL
    # --------------------------------------------------------
    def build_model(self):
        model = Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=self.input_shape),
            MaxPooling2D((2,2)),

            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D((2,2)),

            Conv2D(128, (3,3), activation='relu'),
            MaxPooling2D((2,2)),

            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        return model

    # --------------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------------
    def train(self, train_images, train_labels, epochs=10, batch_size=32):
        if self.model is None:
            self.build_model()

        # Convert to numpy array
        train_images = np.array(train_images).reshape(-1, 64, 64, 1) / 255.0
        train_labels = np.array(train_labels)

        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
        )
        datagen.fit(train_images)

        history = self.model.fit(
            datagen.flow(train_images, train_labels, batch_size=batch_size),
            epochs=epochs,
            verbose=1
        )

        return history

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------
    def save(self, path="thermal_cnn_model.keras"):
        if self.model:
            self.model.save(path)
            print(f"Model saved to {path}")
        else:
            print("Model is not built yet.")

    # --------------------------------------------------------
    # PREDICT ANOMALY TYPE
    # --------------------------------------------------------
    def predict(self, img):
        img = np.array(img).reshape(1, 64, 64, 1) / 255.0
        preds = self.model.predict(img)[0]
        label = np.argmax(preds)
        confidence = float(preds[label])
        return label, confidence