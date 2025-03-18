import os
import librosa
import numpy as np

DATASET_PATH = "../data/GTZAN/"
OUTPUT_PATH = "../data/preprocessed/"

GENRES = ["blues", "classical", "country", "disco", "hiphop", 
          "jazz", "metal", "pop", "reggae", "rock"]

# Create output directory
os.makedirs(OUTPUT_PATH, exist_ok=True)

X, y = [], []

for genre_idx, genre in enumerate(GENRES):
    genre_folder = os.path.join(DATASET_PATH, genre)
    for file in os.listdir(genre_folder):
        if file.endswith(".wav"):  # GTZAN uses WAV files
            file_path = os.path.join(genre_folder, file)
            y_audio, sr = librosa.load(file_path, sr=22050, mono=True)
            
            # Convert to Mel spectrogram
            spectrogram = librosa.feature.melspectrogram(y=y_audio, sr=sr, n_mels=128)
            spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
            
            # Resize to (128,128)
            spectrogram_resized = np.resize(spectrogram_db, (128, 128))
            
            X.append(spectrogram_resized)
            y.append(genre_idx)  # Store genre index

# Convert to NumPy arrays and save
X = np.array(X).reshape(-1, 128, 128, 1)  # Add channel dimension
y = np.array(y)

np.save(os.path.join(OUTPUT_PATH, "X.npy"), X)
np.save(os.path.join(OUTPUT_PATH, "y.npy"), y)

print("Preprocessing completed. Data saved in:", OUTPUT_PATH)
