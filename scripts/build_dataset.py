# scripts/build_dataset.py
import pandas as pd
from _paths import RAW_DIR, PROCESSED_DIR


def main():
    news_path = RAW_DIR / "news_raw.csv"
    weibo_path = RAW_DIR / "weibo_raw.csv"

    dfs = []

    # ----------------- 读 news_raw -----------------
    if news_path.exists():
        news_df = pd.read_csv(news_path)

        # 确保必要列存在
        if "country" not in news_df.columns:
            news_df["country"] = "CN"
        if "source" not in news_df.columns:
            news_df["source"] = "news"
        if "source_type" not in news_df.columns:
            news_df["source_type"] = "news"

        # 这几个列名是 crawl_news 里已经保证的：title, date, content, url
        dfs.append(news_df)
    else:
        print("WARNING: news_raw.csv not found at", news_path)

    # ----------------- 读 weibo_raw（如存在） -----------------
    if weibo_path.exists():
        weibo_df = pd.read_csv(weibo_path)

        # weibo_raw 里已经有 country / source / source_type
        if "country" not in weibo_df.columns:
            weibo_df["country"] = "CN"
        if "source" not in weibo_df.columns:
            weibo_df["source"] = "weibo"
        if "source_type" not in weibo_df.columns:
            weibo_df["source_type"] = "social"

        dfs.append(weibo_df)
    else:
        print("INFO: weibo_raw.csv not found – will only use news.")

    if not dfs:
        print("No input data, exit.")
        return

    # ----------------- 合并 -----------------
    all_df = pd.concat(dfs, ignore_index=True)

    # 如果微博没有 title / url，先创建空列；news 那边已经有
    for col in ["title", "url"]:
        if col not in all_df.columns:
            all_df[col] = ""

    # 如果有 user_name，就用它给微博行填一个“标题”
    if "user_name" in all_df.columns:
        mask = (
            (all_df["source"].astype(str) == "weibo")
            & (all_df["title"].isna() | (all_df["title"].astype(str) == ""))
        )
        all_df.loc[mask, "title"] = all_df.loc[mask, "user_name"].astype(str)

    # 统一填充缺失值，避免后面出现 NaN
    for col in ["country", "source", "source_type", "date", "title", "content", "url"]:
        if col not in all_df.columns:
            all_df[col] = ""
        all_df[col] = all_df[col].astype(str).fillna("")

    # 只保留统一后的字段
    all_df = all_df[
        ["country", "source", "source_type", "date", "title", "content", "url"]
    ]

    # 删掉内容太短的（比如新闻页抓错只有几个字的）
    all_df = all_df[all_df["content"].astype(str).str.len() > 50].reset_index(
        drop=True
    )

    out_path = PROCESSED_DIR / "all_texts.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("Saved unified dataset to", out_path)
    print(all_df.head())


if __name__ == "__main__":
    main()
