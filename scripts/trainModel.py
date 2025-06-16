import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

DATA_PATH = "../bvAnalyser/data/preprocessed/"

# Load preprocessed data
X = np.load(os.path.join(DATA_PATH, "X.npy"))
y = np.load(os.path.join(DATA_PATH, "y.npy"))

# Reshape X to include a channel dimension for CNN
X = X[..., np.newaxis]  # Shape becomes (samples, features, 1)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Compute class weights
class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# Define CNN model with improvements
model = keras.Sequential([
    layers.Conv1D(64, kernel_size=3, activation="relu", input_shape=(X.shape[1], 1)),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(128, kernel_size=3, activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(256, kernel_size=3, activation="relu"),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling1D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(len(np.unique(y)), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

early_stopping = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), class_weight=class_weights, callbacks=[early_stopping])

# Save model
os.makedirs("../bvAnalyser/models/", exist_ok=True)
model.save("../bvAnalyser/models/model.h5")

print("Training completed with augmented data. Model saved as model.h5")
