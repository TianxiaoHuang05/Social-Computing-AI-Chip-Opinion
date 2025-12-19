import pandas as pd
df = pd.read_csv("data/processed/all_with_clusters.csv")

cn = df[df["country"] == "CN"]
print(cn["cluster"].value_counts())
print(cn.groupby("source_type")["cluster"].value_counts(normalize=True))
