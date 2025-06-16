import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

CSV_PATH = "../bvAnalyser/data/features_30_sec.csv"
OUTPUT_PATH = "../bvAnalyser/data/preprocessed/"

os.makedirs(OUTPUT_PATH, exist_ok=True)

data = pd.read_csv(CSV_PATH)

X = data.iloc[:, 1:-1].values  # All columns exc filename and label
y = data.iloc[:, -1].values    # Label column

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

np.save(os.path.join(OUTPUT_PATH, "label_classes.npy"), label_encoder.classes_)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Data augmentation: Add noise to features
def augment_data(X, noise_factor=0.01):
    noise = np.random.normal(0, noise_factor, X.shape)
    return X + noise

X_augmented = augment_data(X)
y_augmented = y  # Labels remain unchanged

# Combine original and augmented data
X_combined = np.vstack([X, X_augmented])
y_combined = np.hstack([y, y_augmented])

np.save(os.path.join(OUTPUT_PATH, "X.npy"), X_combined)
np.save(os.path.join(OUTPUT_PATH, "y.npy"), y_combined)

print("Preprocessing completed with augmentation. Data saved in:", OUTPUT_PATH)
