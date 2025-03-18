import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


import kagglehub

# Download latest version
path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")

print("Path to dataset files:", path)


# Vytvoreni jednoduche CNN
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")  # 10 žánrů
])

# Kompilace modelu
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])



# Trénování modelu (data pripravit)
# model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
