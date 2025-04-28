import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os

DATA_PATH = "../bvAnalyser/data/preprocessed/"

# Load preprocessed data
X = np.load(os.path.join(DATA_PATH, "X.npy"))
y = np.load(os.path.join(DATA_PATH, "y.npy"))

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(X.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Save the model
os.makedirs("../bvAnalyser/models/", exist_ok=True)
model.save("../bvAnalyser/models/model.h5")

print("Training completed. Model saved as model.h5")
