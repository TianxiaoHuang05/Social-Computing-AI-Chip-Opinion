# scripts/us_topic_mini.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from _paths import PROCESSED_DIR


def main():
    df = pd.read_csv(PROCESSED_DIR / "all_texts_clean.csv")

    us_news = df[(df["country"] == "US") & (df["source_type"] == "news")].copy()
    print(f"美国新闻条数: {len(us_news)}")
    if len(us_news) == 0:
        print("没有筛到美国新闻，检查一下 country 和 source_type 字段。")
        return

    texts = us_news["tokens"].fillna("").tolist()

    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        stop_words='english',  # 关键改这里
    )
    X = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()

    k = 2
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    us_news["us_topic"] = kmeans.fit_predict(X)

    def print_topic_top_terms(topic_id, topn=15):
        centroid = kmeans.cluster_centers_[topic_id]
        top_idx = centroid.argsort()[::-1][:topn]
        top_terms = [terms[i] for i in top_idx]
        print(f"\nTopic {topic_id} top terms:")
        print(", ".join(top_terms))

    for t in range(k):
        print_topic_top_terms(t)

    out_path = PROCESSED_DIR / "us_news_topics.csv"
    us_news.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("\n已保存美国新闻聚类结果到：", out_path)


if __name__ == "__main__":
    main()
