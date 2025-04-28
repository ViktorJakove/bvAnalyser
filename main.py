import os

#csv gen
print("converting your song...")
os.system("python scripts/csvGeneration.py")

#testing
print("enumerating the prediction...")
os.system("python scripts/testSong.py")
