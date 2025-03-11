import librosa
import numpy as np
import librosa.display
import tensorflow as tf
import matplotlib.pyplot as plt

# Načtení modelu
model = tf.keras.models.load_model("model.h5")

# Načtení testovací skladby
y, sr = librosa.load("test_song.mp3", sr=22050)

# Vytvoření spektrogramu
spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

# Vizualizace spektrogramu
plt.figure(figsize=(10, 4))
librosa.display.specshow(spectrogram_db, sr=sr, x_axis="time", y_axis="mel")
plt.colorbar(format="%+2.0f dB")
plt.title("Testovací Mel-Spectrogram")
plt.show()

# Převod do správného tvaru pro model
spectrogram_resized = np.expand_dims(spectrogram_db, axis=[0, -1])

# Predikce modelu
prediction = model.predict(spectrogram_resized)
predicted_label = np.argmax(prediction)

print(f"Predikovaný žánr: {predicted_label}")