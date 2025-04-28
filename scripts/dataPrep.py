import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

CSV_PATH = "../bvAnalyser/data/features_30_sec.csv"
OUTPUT_PATH = "../bvAnalyser/data/preprocessed/"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# Read the CSV file
data = pd.read_csv(CSV_PATH)

# Extract features and labels
X = data.iloc[:, 1:-1].values  # All columns except filename and label
y = data.iloc[:, -1].values    # Label column

# Encode labels as integers
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Save the label encoder classes for later use
np.save(os.path.join(OUTPUT_PATH, "label_classes.npy"), label_encoder.classes_)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save the processed data
np.save(os.path.join(OUTPUT_PATH, "X.npy"), X)
np.save(os.path.join(OUTPUT_PATH, "y.npy"), y)

print("Preprocessing completed. Data saved in:", OUTPUT_PATH)
