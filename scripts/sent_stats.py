# scripts/sent_stats.py
import pandas as pd
from _paths import PROCESSED_DIR

df = pd.read_csv(PROCESSED_DIR / "all_with_sentiment.csv")

def show_dist(name, sub):
    print(f"\n{name}")
    print(sub["sentiment_label"].value_counts())
    print(sub["sentiment_label"].value_counts(normalize=True))

# 中国整体
cn = df[df["country"] == "CN"]
show_dist("CN overall", cn)

# 美国整体
us = df[df["country"] == "US"]
show_dist("US overall", us)

# 中国：新闻 vs 微博
cn_news = cn[cn["source_type"] == "news"]
cn_social = cn[cn["source_type"] == "social"]
show_dist("CN news", cn_news)
show_dist("CN social", cn_social)
