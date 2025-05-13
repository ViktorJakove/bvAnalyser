import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd
import librosa
import music21
import warnings


MODEL_PATH = "../bvAnalyser/models/model.h5"
TEST_FILE = "../bvAnalyser/data/custom_features_30_sec.csv"
LABEL_CLASSES_PATH = "../bvAnalyser/data/preprocessed/label_classes.npy"
AUDIO_DIR = "../bvAnalyser/data/custom_audio/"


audio_files = [f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(".wav")]
if not audio_files:
    raise FileNotFoundError(f"No .wav files found in {AUDIO_DIR}")

audio_path = os.path.join(AUDIO_DIR, audio_files[0])

if not os.path.exists(LABEL_CLASSES_PATH):
    raise FileNotFoundError(f"Label classes file not found at {LABEL_CLASSES_PATH}. Please run dataprep.py to generate it.")

model = tf.keras.models.load_model(MODEL_PATH)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
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

y, sr = librosa.load(audio_path, sr=None)
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
tempo_rounded = int(np.round(tempo[0]))

chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
chroma_vector = np.mean(chroma_cq, axis=1)

stream = music21.stream.Stream()
for i, weight in enumerate(chroma_vector):
    n = music21.note.Note()
    n.pitch.pitchClass = i
    n.quarterLength = max(min(weight * 4, 4), 0.01)
    stream.append(n)

key_obj = stream.analyze('key')

print("\nPrediction for the song:")
print(f"File analyzed: {audio_files[0]}")
print(f"Predicted Genre: {predicted_genre}")
print(f"Tempo of the song: {tempo_rounded} BPM")
print(f"Key of the song: {key_obj.tonic.name} {key_obj.mode.capitalize()}")