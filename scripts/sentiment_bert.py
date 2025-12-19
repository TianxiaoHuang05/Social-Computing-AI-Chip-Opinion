# scripts/sentiment_bert.py
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

from _paths import PROCESSED_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ====== 根据自己需要换成别的模型 ====== #
CH_MODEL_NAME = "uer/roberta-base-finetuned-jd-binary-chinese"
EN_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
# ================================== #

print("Loading CN model:", CH_MODEL_NAME)
ch_tokenizer = AutoTokenizer.from_pretrained(CH_MODEL_NAME)
ch_model = AutoModelForSequenceClassification.from_pretrained(CH_MODEL_NAME).to(device)

print("Loading EN model:", EN_MODEL_NAME)
en_tokenizer = AutoTokenizer.from_pretrained(EN_MODEL_NAME)
en_model = AutoModelForSequenceClassification.from_pretrained(EN_MODEL_NAME).to(device)


def predict_sentiment(texts, tokenizer, model):
    """
    返回 [(label, prob), ...]
    label 一般 0=负向, 1=正向（具体看模型文档）
    """
    results = []
    model.eval()
    for t in tqdm(texts, desc="Sentiment predicting"):
        if not isinstance(t, str) or t.strip() == "":
            results.append((None, None))
            continue
        with torch.no_grad():
            inputs = tokenizer(
                t,
                truncation=True,
                padding="max_length",
                max_length=256,
                return_tensors="pt",
            ).to(device)
            logits = model(**inputs).logits
            prob = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            label = int(prob.argmax())
            results.append((label, float(prob[label])))
    return results


def main():
    in_path = PROCESSED_DIR / "all_texts_clean.csv"
    df = pd.read_csv(in_path)

    cn_mask = df["country"].str.upper() == "CN"
    us_mask = df["country"].str.upper() == "US"

    cn_texts = df.loc[cn_mask, "clean_content"].fillna("").tolist()
    us_texts = df.loc[us_mask, "clean_content"].fillna("").tolist()

    print("CN texts:", len(cn_texts), "US texts:", len(us_texts))

    cn_res = predict_sentiment(cn_texts, ch_tokenizer, ch_model)
    us_res = predict_sentiment(us_texts, en_tokenizer, en_model)

    df.loc[cn_mask, ["sentiment_label", "sentiment_conf"]] = cn_res
    df.loc[us_mask, ["sentiment_label", "sentiment_conf"]] = us_res

    out_path = PROCESSED_DIR / "all_with_sentiment.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("Saved sentiment results to", out_path)

    # 你可以顺手看一下中美情绪分布
    print(df.groupby("country")["sentiment_label"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
