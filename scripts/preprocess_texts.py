# scripts/preprocess_texts.py
import re

import jieba
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize

from _paths import PROCESSED_DIR

# 第一次跑需要下载 punkt，后面如果已经有就不会再下
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


def basic_clean(text: str) -> str:
    """非常基础的清洗：去 URL、@、#话题、HTML 实体、多余空白"""
    if not isinstance(text, str):
        text = str(text)

    # 去 URL
    text = re.sub(r"http[s]?://\S+", " ", text)
    # 去 @xxx 和 #话题#
    text = re.sub(r"[@#]\S+", " ", text)
    # 英文 html 实体 &nbsp; 等
    text = re.sub(r"&[a-z]+;", " ", text)
    # 合并空白
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_cn(text: str) -> str:
    words = jieba.lcut(text)
    return " ".join(words)


def tokenize_en(text: str) -> str:
    words = word_tokenize(text)
    return " ".join(words)


def main():
    in_path = PROCESSED_DIR / "all_texts.csv"
    df = pd.read_csv(in_path)

    # 保证有 content / country 两列
    if "content" not in df.columns:
        raise ValueError("all_texts.csv 缺少 'content' 列")
    if "country" not in df.columns:
        # 没有就默认都按中文处理
        df["country"] = "CN"

    # 统一转成字符串，去掉 NaN
    df["content"] = df["content"].astype(str).fillna("")
    df["country"] = df["country"].astype(str).fillna("")

    cleaned = []
    tokens = []

    for content, country in zip(df["content"], df["country"]):
        txt = basic_clean(content)
        # 如果清洗后完全空，就直接记空
        if not txt:
            cleaned.append("")
            tokens.append("")
            continue

        if country.upper() == "CN":
            tok = tokenize_cn(txt)
        else:
            tok = tokenize_en(txt)

        cleaned.append(txt)
        tokens.append(tok)

    df["clean_content"] = cleaned
    df["tokens"] = tokens

    # 删掉清洗后仍然是空的行
    df = df[df["clean_content"] != ""].reset_index(drop=True)

    out_path = PROCESSED_DIR / "all_texts_clean.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("Saved cleaned texts to", out_path)
    print(df[["country", "clean_content", "tokens"]].head())


if __name__ == "__main__":
    main()
