import librosa
import numpy as np
import librosa.display
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

# Načtení modelu
model = tf.keras.models.load_model("model.h5")

# Načtení testovací skladby
y, sr = librosa.load("test_song.mp3", sr=22050, mono=True)

# Vytvoření spektrogramu
spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)  # Ensure it has 128 mel bands
spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

# Vizualizace spektrogramu
plt.figure(figsize=(10, 4))
librosa.display.specshow(spectrogram_db, sr=sr, x_axis="time", y_axis="mel")
plt.colorbar(format="%+2.0f dB")
plt.title("Testovací Mel-Spectrogram")
plt.show()

# Převod do správného tvaru pro model
spectrogram_resized = cv2.resize(spectrogram_db, (128, 128))  # Resize to match model input
spectrogram_resized = np.expand_dims(spectrogram_resized, axis=[0, -1])  # Přidání dimenzí (batch, channel)

# Normalizace na rozsah [0, 1]
spectrogram_resized = (spectrogram_resized - spectrogram_resized.min()) / (spectrogram_resized.max() - spectrogram_resized.min())

# Predikce modelu
prediction = model.predict(spectrogram_resized)
predicted_label = np.argmax(prediction)

# Mapování indexu na žánr
genre_mapping = {0: "Blues", 1: "Classical", 2: "Country", 3: "Disco", 4: "Hip-Hop",
                 5: "Jazz", 6: "Metal", 7: "Pop", 8: "Reggae", 9: "Rock"}

print(f"Predikovaný žánr: {genre_mapping.get(predicted_label, 'Neznámý žánr')}")