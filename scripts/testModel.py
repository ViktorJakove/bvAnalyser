import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa.display
import cv2

MODEL_PATH = "../models/model.h5"
TEST_FILE = "../data/test_song.mp3"

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Load test song
y, sr = librosa.load(TEST_FILE, sr=22050, mono=True)

# Convert to Mel spectrogram
spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

# Resize spectrogram to 128x128
spectrogram_resized = cv2.resize(spectrogram_db, (128, 128))
spectrogram_resized = np.expand_dims(spectrogram_resized, axis=[0, -1])

# Normalize to [0,1]
spectrogram_resized = (spectrogram_resized - spectrogram_resized.min()) / (spectrogram_resized.max() - spectrogram_resized.min())

# Predict
prediction = model.predict(spectrogram_resized)
predicted_label = np.argmax(prediction)

# Genre mapping
genre_mapping = {0: "Blues", 1: "Classical", 2: "Country", 3: "Disco", 4: "Hip-Hop",
                 5: "Jazz", 6: "Metal", 7: "Pop", 8: "Reggae", 9: "Rock"}

print(f"Predicted Genre: {genre_mapping.get(predicted_label, 'Unknown Genre')}")
