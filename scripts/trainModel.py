import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os

DATA_PATH = "../data/preprocessed/"

X = np.load(os.path.join(DATA_PATH, "X.npy"))
y = np.load(os.path.join(DATA_PATH, "y.npy"))

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

os.makedirs("../models/", exist_ok=True)
model.save("../models/model.h5")

print("Training completed. Model saved as model.h5")
