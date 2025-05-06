import os
import librosa
import pandas as pd
import numpy as np

AUDIO_FOLDER = "../bvAnalyser/data/custom_audio"
OUTPUT_CSV = "../bvAnalyser/data/custom_features_30_sec.csv"

GENRE_MAPPING = {
    "pop": "Pop",
    "rock": "Rock",
    "jazz": "Jazz",
    "classical": "Classical",
    "hiphop": "Hip-Hop",
    "blues": "Blues",
    "country": "Country",
    "reggae": "Reggae",
}

def extract_features(file_path):
    try:
        print(f"Processing file: {file_path}")
        
        y, sr = librosa.load(file_path, sr=None)

        target_length = sr * 30  # 30 sec
        if len(y) < target_length:
            padding = target_length - len(y)
            y = np.pad(y, (0, padding), mode='constant')
        else:
            start_sample = (len(y) - target_length) // 2
            end_sample = start_sample + target_length
            y = y[start_sample:end_sample]
        try:
            tempo = float(librosa.feature.rhythm.tempo(y=y, sr=sr)[0])
        except AttributeError:
            tempo = float(librosa.beat.tempo(y=y, sr=sr)[0])

        features = {
            "filename": os.path.basename(file_path),
            "length": len(y),
            "chroma_stft_mean": np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
            "chroma_stft_var": np.var(librosa.feature.chroma_stft(y=y, sr=sr)),
            "rms_mean": np.mean(librosa.feature.rms(y=y)),
            "rms_var": np.var(librosa.feature.rms(y=y)),
            "spectral_centroid_mean": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            "spectral_centroid_var": np.var(librosa.feature.spectral_centroid(y=y, sr=sr)),
            "spectral_bandwidth_mean": np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
            "spectral_bandwidth_var": np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
            "rolloff_mean": np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
            "rolloff_var": np.var(librosa.feature.spectral_rolloff(y=y, sr=sr)),
            "zero_crossing_rate_mean": np.mean(librosa.feature.zero_crossing_rate(y)),
            "zero_crossing_rate_var": np.var(librosa.feature.zero_crossing_rate(y)),
            "harmony_mean": np.mean(librosa.effects.harmonic(y)),
            "harmony_var": np.var(librosa.effects.harmonic(y)),
            "perceptr_mean": np.mean(librosa.effects.percussive(y)),
            "perceptr_var": np.var(librosa.effects.percussive(y)),
            "tempo": tempo,
        }

        # genre (filename)
        filename = os.path.basename(file_path).lower()
        assigned_genre = "Unknown"
        for keyword, genre in GENRE_MAPPING.items():
            if keyword in filename:
                assigned_genre = genre
                break
        features["label"] = assigned_genre

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(1, 21):
            features[f"mfcc{i}_mean"] = np.mean(mfccs[i - 1])
            features[f"mfcc{i}_var"] = np.var(mfccs[i - 1])

        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# vsechnio
data = []
for file in os.listdir(AUDIO_FOLDER):
    if file.endswith(".wav"):  # Process only .wav files
        file_path = os.path.join(AUDIO_FOLDER, file)
        features = extract_features(file_path)
        if features:
            data.append(features)

# CSV file
df = pd.DataFrame(data)
df.to_csv(OUTPUT_CSV, index=False)

print(f"Feature extraction completed. CSV saved to {OUTPUT_CSV}")