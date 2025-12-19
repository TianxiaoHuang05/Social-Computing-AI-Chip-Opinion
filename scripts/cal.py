import pandas as pd
df = pd.read_csv("data/processed/all_texts.csv")

print(df["country"].value_counts())
print(df["source_type"].value_counts())
print(df.groupby(["country", "source_type"]).size())
