import os

#preprocessing
print("preprocessing...")
os.system("python scripts/dataPrep.py")

#training
print("model training...")
os.system("python scripts/trainModel.py")

#testing
print("model testing...")
os.system("python scripts/testModel.py")