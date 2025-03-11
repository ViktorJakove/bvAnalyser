import librosa
import librosa.display
import matplotlib.pyplot as plt

# Načtení skladby
y, sr = librosa.load("", sr=22050)

# Vytvoření spektrogramu
spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

# Zobrazení spektrogramu
plt.figure(figsize=(10, 4))
librosa.display.specshow(spectrogram_db, sr=sr, x_axis="time", y_axis="mel")
plt.colorbar(format="%+2.0f dB")
plt.title("Mel-spectrogram")
plt.show()
