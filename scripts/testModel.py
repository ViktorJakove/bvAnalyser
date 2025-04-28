import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

MODEL_PATH = "../bvAnalyser/models/model.h5"
TEST_FILE = "../bvAnalyser/data/custom_features_30_sec.csv"

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# Load test data
data = pd.read_csv(TEST_FILE)

# Extract features and labels
X_test = data.drop(columns=["filename", "label"]).values  # Drop non-numeric columns
y_test = data["label"].values  # Extract the label column

# Normalize features
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

# Ensure the feature vector has 58 features
if X_test.shape[1] < 58:
    missing_features = 58 - X_test.shape[1]
    X_test = np.hstack([X_test, np.zeros((X_test.shape[0], missing_features))])

# Encode labels
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(["class1", "class2", "class3"])  # Replace with actual class names

# Filter out rows with invalid labels
valid_classes = set(label_encoder.classes_)
data = data[data["label"].isin(valid_classes)]
y_test = data["label"].values

# Encode labels
y_test = label_encoder.transform(y_test)

# Make predictions
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

# Print raw predictions and predicted labels
print("\nRaw Predictions (Probabilities):")
print(predictions)

print("\nPredicted Labels (Indices):")
print(predicted_labels)

# Map predictions to genre names
genre_mapping = dict(enumerate(label_encoder.classes_))
predicted_genres = [genre_mapping[label] for label in predicted_labels]

print("\nPredicted Genres:")
print(predicted_genres)

# Compute statistics
print("\nClassification Report:")
print(classification_report(y_test, predicted_labels, labels=range(len(label_encoder.classes_)), target_names=label_encoder.classes_))

print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, predicted_labels)
print(conf_matrix)

# Compute per-genre statistics
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
