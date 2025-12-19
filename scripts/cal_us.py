# scripts/cal_us.py
import pandas as pd
from _paths import PROCESSED_DIR

# 1. 读聚类结果
df_cluster = pd.read_csv(PROCESSED_DIR / "all_with_clusters.csv")

us = df_cluster[df_cluster["country"] == "US"].copy()
print("美国新闻条数：", len(us))

print("\n【按 cluster 计数】")
print(us["cluster"].value_counts().sort_index())

print("\n【按 cluster 占比】")
print(us["cluster"].value_counts(normalize=True).sort_index())

# 如果你已经跑过 sentiment_bert.py，就再读情感结果
try:
    df_sent = pd.read_csv(PROCESSED_DIR / "all_with_sentiment.csv")
    us_sent = df_sent[df_sent["country"] == "US"].copy()
    print("\n【情感 label 分布】")
    print(us_sent["sentiment_label"].value_counts())
    print("\n【情感 label 占比】")
    print(us_sent["sentiment_label"].value_counts(normalize=True))
except FileNotFoundError:
    print("\n还没有 all_with_sentiment.csv，就先不算情感分布。")