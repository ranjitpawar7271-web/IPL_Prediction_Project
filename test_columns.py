import pandas as pd

df = pd.read_csv("data/raw/IPL.csv")

print("Columns in dataset:\n")
print(df.columns.tolist())