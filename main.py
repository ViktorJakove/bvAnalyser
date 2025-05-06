import os
from pathlib import Path

files_to_check = [
    "data/preprocessed",
    "data/preprocessed/label_classes.npy",
    "data/preprocessed/X.npy",
    "data/preprocessed/Y.npy"
]

def check_files():
    for file_path in files_to_check:
        path = Path(file_path)
        if not path.exists() or not path.is_dir() and not path.is_file():
            print("running data preparation script...")
            os.system("python scripts/dataPrep.py")

check_files()

#csv gen
print("converting your song...")
os.system("python scripts/csvGeneration.py")

#testing
print("enumerating the prediction...")
os.system("python scripts/testSong.py")
