import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

MODEL_PATH = "../bvAnalyser/models/model.h5"
TEST_FILE = "../bvAnalyser/data/custom_features_30_sec.csv"
LABEL_CLASSES_PATH = "../bvAnalyser/data/preprocessed/label_classes.npy"

model = tf.keras.models.load_model(MODEL_PATH)

data = pd.read_csv(TEST_FILE)

X_test = data.drop(columns=["filename", "label"]).values
y_test = data["label"].values

scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

if X_test.shape[1] < 58:
    missing_features = 58 - X_test.shape[1]
    X_test = np.hstack([X_test, np.zeros((X_test.shape[0], missing_features))])

label_classes = np.load(LABEL_CLASSES_PATH, allow_pickle=True)

label_encoder = LabelEncoder()
label_encoder.classes_ = label_classes

valid_classes = set(label_encoder.classes_)
data = data[data["label"].isin(valid_classes)] 
X_test = data.drop(columns=["filename", "label"]).values
y_test = data["label"].values

y_test = label_encoder.transform(y_test)

if X_test.shape[0] == 0:
    print("\nNo valid samples in the test dataset. Skipping predictions.")
else:
    predictions = model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)

    print("\nRaw Predictions (Probabilities):")
    print(predictions)

    print("\nPredicted Labels (Indices):")
    print(predicted_labels)

    genre_mapping = dict(enumerate(label_encoder.classes_))
    predicted_genres = [genre_mapping.get(label, f"Unknown (class index {label})") for label in predicted_labels]

    print("\nPredicted Genres:")
    print(predicted_genres)

    if len(y_test) > 0:
        print("\nClassification Report:")
        print(classification_report(y_test, predicted_labels, labels=range(len(label_encoder.classes_)), target_names=label_encoder.classes_))

        print("\nConfusion Matrix:")
        conf_matrix = confusion_matrix(y_test, predicted_labels)
        print(conf_matrix)

        genre_stats = {}
        for genre_idx, genre_name in enumerate(label_encoder.classes_):
            genre_mask = y_test == genre_idx
            total_samples = genre_mask.sum()
            correct_predictions = (predicted_labels[genre_mask] == genre_idx).sum()
            accuracy = correct_predictions / total_samples if total_samples > 0 else 0
            genre_stats[genre_name] = {
                "total_samples": total_samples,
                "correct_predictions": correct_predictions,
                "accuracy": accuracy,
            }

        print("\nPer-Genre Statistics:")
        for genre, stats in genre_stats.items():
            print(f"{genre}: {stats}")
