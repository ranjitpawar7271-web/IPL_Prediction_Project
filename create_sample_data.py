import pandas as pd

df = pd.read_csv("data/raw/IPL.csv")

df_small = df.sample(frac=0.2, random_state=42)

df_small.to_csv("data/raw/IPL_small.csv", index=False)
