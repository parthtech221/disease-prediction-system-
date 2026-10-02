import pandas as pd


df = pd.read_csv("dataset/disease_symptom.csv")

# Quick look at the data
print(df.head())
print(df.columns)
print(df.shape)
