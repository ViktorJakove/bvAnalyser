import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd
import librosa
import librosa.display

MODEL_PATH = "../bvAnalyser/models/model.h5"
TEST_FILE = "../bvAnalyser/data/custom_features_30_sec.csv"
LABEL_CLASSES_PATH = "../bvAnalyser/data/preprocessed/label_classes.npy"

if not os.path.exists(LABEL_CLASSES_PATH):
    raise FileNotFoundError(f"Label classes file not found at {LABEL_CLASSES_PATH}. Please run dataprep.py to generate it.")

model = tf.keras.models.load_model(MODEL_PATH)

data = pd.read_csv(TEST_FILE)

X_test = data.drop(columns=["filename", "label"]).values

scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

if X_test.shape[1] < 58:
    missing_features = 58 - X_test.shape[1]
    X_test = np.hstack([X_test, np.zeros((X_test.shape[0], missing_features))])

label_classes = np.load(LABEL_CLASSES_PATH, allow_pickle=True)

predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

predicted_genre = label_classes[predicted_labels[0]]

y, sr = librosa.load("../bvAnalyser/data/custom_audio/A. Dvorak_ Slavonic dances No.3, Polka, A flat major(1).wav", sr=None)
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

chroma = librosa.feature.chroma_cens(y=y, sr=sr)
key_index = np.argmax(np.mean(chroma, axis=1))
keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
key = keys[key_index]

print("\nPrediction for the song:")
print(f"Predicted Genre: {predicted_genre}")
print(f"Tempo of the song: {tempo} BPM")
print(f"Key of the song: {key}")
