# scripts/merge_weibo.py
from glob import glob

import pandas as pd

from _paths import RAW_DIR, PROJECT_ROOT


def merge_weibo():
    """
    合并 weibo_output 下所有 csv，并抽取：
    - date      : 微博发布时间
    - user_name : 用户昵称 / 微博作者
    - content   : 微博正文 / 微博内容

    最终输出列：
    country, source, source_type, date, user_name, content
    """
    weibo_dir = PROJECT_ROOT / "weibo_output"
    files = glob(str(weibo_dir / "*.csv"))

    if not files:
        print("No weibo csv found in", weibo_dir, "— you can skip this step.")
        return

    dfs = []
    for f in files:
        try:
            # 默认读取，如果报错再换编码
            try:
                df = pd.read_csv(f)
            except Exception:
                df = pd.read_csv(f, encoding="utf-8-sig", errors="ignore")
            print("Loaded", f, "columns:", list(df.columns))
            dfs.append(df)
        except Exception as e:
            print("Error reading", f, e)

    if not dfs:
        print("No valid weibo csv.")
        return

    weibo_df = pd.concat(dfs, ignore_index=True)

    # ---------- 根据实际列名做更宽松的匹配 ---------- #
    rename_map = {}

    for col in weibo_df.columns:
        col_str = str(col)
        low = col_str.lower()

        # 1) 正文 / 内容列
        # 典型：content, 微博内容, 微博正文, 正文, 内容
        if (
            low == "text"
            or low == "content"
            or "content" in low
            or "微博内容" in col_str
            or "微博正文" in col_str
            or (col_str == "正文")
            or (col_str.endswith("内容") and "转发" not in col_str)  # 排除“转发内容”之类也没关系，其实也算内容
        ):
            rename_map[col] = "content"
            continue

        # 2) 时间 / 日期列
        # 典型：created_at, publish_time, 发布时间, 发表时间
        if (
            low == "created_at"
            or low == "publish_time"
            or "time" in low
            or "date" in low
            or "发布时间" in col_str
            or "发表时间" in col_str
            or col_str == "时间"
        ):
            rename_map[col] = "date"
            continue

        # 3) 用户名 / 昵称列
        # 典型：user_name, username, screen_name, 用户昵称, 用户名, 微博作者, 博主昵称
        if (
            "user_name" in low
            or "username" in low
            or "screen_name" in low
            or "用户昵称" in col_str
            or ("昵称" in col_str and "id" not in low)
            or "微博作者" in col_str
            or "博主昵称" in col_str
            or "作者名" in col_str
        ):
            rename_map[col] = "user_name"
            continue

    print("rename_map =", rename_map)
    weibo_df = weibo_df.rename(columns=rename_map)

    # ---------- 补充来源信息 ---------- #
    weibo_df["country"] = "CN"
    weibo_df["source"] = "weibo"
    weibo_df["source_type"] = "social"

    # 确保三列存在（如果真没匹配到就留空）
    for c in ["date", "user_name", "content"]:
        if c not in weibo_df.columns:
            print(f"[warn] column '{c}' not found, filled with empty string.")
            weibo_df[c] = ""

    cols = ["country", "source", "source_type", "date", "user_name", "content"]
    weibo_df = weibo_df[cols]

    out_path = RAW_DIR / "weibo_raw.csv"
    weibo_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("Saved merged weibo to", out_path)
    print(weibo_df.head())


def main():
    merge_weibo()


if __name__ == "__main__":
    main()
