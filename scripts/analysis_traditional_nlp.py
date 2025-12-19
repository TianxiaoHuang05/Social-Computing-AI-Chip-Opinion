# scripts/analysis_traditional_nlp.py
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

from _paths import PROCESSED_DIR, FIG_DIR


def main():
    in_path = PROCESSED_DIR / "all_texts_clean.csv"
    df = pd.read_csv(in_path)

    texts = df["tokens"].fillna("").tolist()

    # TF-IDF 向量
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)

    # 聚成 4 类，你可以按需要改 k
    k = 4
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X)

    terms = vectorizer.get_feature_names_out()

    def print_cluster_top_terms(cluster_id, topn=20):
        centroid = kmeans.cluster_centers_[cluster_id]
        top_idx = centroid.argsort()[::-1][:topn]
        top_terms = [terms[i] for i in top_idx]
        print(f"\nCluster {cluster_id} top terms:")
        print(", ".join(top_terms))

    for c in range(k):
        print_cluster_top_terms(c)

    out_csv = PROCESSED_DIR / "all_with_clusters.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("Saved clustered data to", out_csv)

    # 降维画图（小数据可以 toarray，大规模就不要这么干）
    X_dense = X.toarray()
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_dense)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=df["cluster"], s=8)
    plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("TF-IDF + KMeans Clusters (CN & US opinions)")
    plt.tight_layout()

    fig_path = FIG_DIR / "clusters_pca.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print("Saved cluster figure to", fig_path)


if __name__ == "__main__":
    main()
