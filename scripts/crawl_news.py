# -*- coding: utf-8 -*-
"""
从 cn_urls.txt / us_urls.txt 批量抓取新闻正文，并保存到 data/raw/news_raw.csv

依赖:
    pip install requests beautifulsoup4 lxml pandas tqdm
"""

import re
import time
from typing import Dict
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

from _paths import CONFIG_DIR, RAW_DIR

# ------------------- HTTP 会话与基础工具 -------------------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

session = requests.Session()
session.headers.update(HEADERS)


def fetch_html(url: str, sleep: float = 1.0) -> str:
    """
    带简单重试的 HTML 获取。
    - 只接受 text/html / application/xhtml+xml
    - 遇到 PDF 等二进制内容直接跳过
    """
    for _ in range(3):
        try:
            resp = session.get(url, timeout=10)
            if resp.status_code == 200:
                ctype = resp.headers.get("Content-Type", "")
                if (
                    "text/html" not in ctype
                    and "application/xhtml+xml" not in ctype
                ):
                    print(f"[skip] non-HTML content for {url}: {ctype}")
                    return ""
                text = resp.text
                if sleep > 0:
                    time.sleep(sleep)
                return text
            else:
                print(f"[warn] {url} status {resp.status_code}")
        except Exception as e:
            print(f"[error] Error fetching {url}: {e}")
            time.sleep(2)
    return ""


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_date_generic(soup: BeautifulSoup) -> str:
    """
    兜底日期抽取：
    1) <time> 标签
    2) 常见 <meta> 发布时间字段
    3) class 名里包含 time/date/pubtime 的元素
    4) 在全页文本里用正则匹配 YYYY-MM-DD / 中文日期 / 英文月份
    """
    # 1) <time> 标签
    time_tag = soup.find("time")
    if time_tag:
        dt = time_tag.get("datetime")
        if dt:
            return dt.strip()
        txt = time_tag.get_text(strip=True)
        if txt:
            return txt

    # 2) meta 标签
    for prop in ("article:published_time", "og:published_time", "article:modified_time"):
        meta = soup.find("meta", attrs={"property": prop})
        if meta and meta.get("content"):
            return meta["content"].strip()

    for name in ("pubdate", "publishdate", "publish_time", "ptime", "date", "sailthru.date"):
        meta = soup.find("meta", attrs={"name": name})
        if meta and meta.get("content"):
            return meta["content"].strip()

    # 3) class 名里含 time/date/pubtime 的元素
    class_patterns = (
        "time",
        "date",
        "pubtime",
        "pub_time",
        "publish",
        "article-time",
        "news-time",
        "time-source",
    )
    for pattern in class_patterns:
        tag = soup.find(attrs={"class": re.compile(pattern, re.I)})
        if tag:
            txt = tag.get_text(" ", strip=True)
            if txt:
                return txt

    # 4) 正则匹配整页文本
    full_text = soup.get_text(" ", strip=True)

    # 4.1 YYYY-MM-DD HH:MM:SS / YYYY-MM-DD HH:MM
    m = re.search(
        r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}(日)?\s*\d{0,2}:?\d{0,2}:?\d{0,2}?",
        full_text,
    )
    if m:
        return m.group(0)

    # 4.2 仅日期 YYYY-MM-DD / YYYY年MM月DD日
    m = re.search(r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}(日)?", full_text)
    if m:
        return m.group(0)

    # 4.3 英文月份形式：March 12, 2025
    m = re.search(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{1,2},\s+\d{4}",
        full_text,
    )
    if m:
        return m.group(0)

    return ""


# ------------------- 各网站解析函数（可按需慢慢扩展） -------------------


def parse_people_cn(html: str) -> Dict:
    """人民网正文，一般在 div#rwb_zw / .rm_txt_con 里"""
    soup = BeautifulSoup(html, "lxml")
    title_tag = soup.find("h1") or soup.find("h2")
    date_tag = (
        soup.find("span", class_="publish-time")
        or soup.find("span", class_="date")
        or soup.find("div", class_="sou")
    )
    content_div = (
        soup.find("div", id="rwb_zw")
        or soup.find("div", class_="rm_txt_con")
        or soup.find("div", class_="article")
    )

    title = title_tag.get_text(strip=True) if title_tag else ""
    date = date_tag.get_text(strip=True) if date_tag else ""
    if not date:
        date = extract_date_generic(soup)

    if content_div:
        ps = [p.get_text(" ", strip=True) for p in content_div.find_all("p")]
        content = clean_text(" ".join(ps))
    else:
        content = ""

    return {"title": title, "date": date, "content": content}


def parse_globaltimes_cn(html: str) -> Dict:
    """环球时报中文站正文"""
    soup = BeautifulSoup(html, "lxml")
    title_tag = soup.find("h1")
    date_tag = soup.find("span", class_="pub_time") or soup.find(
        "span", class_="time"
    )
    body_div = (
        soup.find("div", class_="article-content")
        or soup.find("div", class_="artical-content")
    )

    title = title_tag.get_text(strip=True) if title_tag else ""
    date = date_tag.get_text(strip=True) if date_tag else ""
    if not date:
        date = extract_date_generic(soup)

    if body_div:
        ps = [p.get_text(" ", strip=True) for p in body_div.find_all("p")]
        content = clean_text(" ".join(ps))
    else:
        content = ""

    return {"title": title, "date": date, "content": content}


def parse_reuters(html: str) -> Dict:
    """Reuters 新闻正文"""
    soup = BeautifulSoup(html, "lxml")
    title_tag = soup.find("h1")
    time_tag = soup.find("time")
    body_div = soup.find("div", attrs={"data-testid": "article-body"})

    title = title_tag.get_text(strip=True) if title_tag else ""
    if time_tag:
        date = time_tag.get_text(strip=True) or time_tag.get("datetime", "")
    else:
        date = ""
    if not date:
        date = extract_date_generic(soup)

    if body_div:
        ps = [p.get_text(" ", strip=True) for p in body_div.find_all("p")]
        content = clean_text(" ".join(ps))
    else:
        content = ""

    return {"title": title, "date": date, "content": content}


def parse_generic(html: str) -> Dict:
    """兜底：取第一个 <h1> / <title> + 所有 <p> 文本，并尽量猜测日期"""
    soup = BeautifulSoup(html, "lxml")
    title_tag = soup.find("h1") or soup.find("title")
    ps = [p.get_text(" ", strip=True) for p in soup.find_all("p")]

    title = title_tag.get_text(strip=True) if title_tag else ""
    content = clean_text(" ".join(ps)) if ps else ""
    date = extract_date_generic(soup)

    return {"title": title, "date": date, "content": content}


def parse_article(url: str, html: str) -> Dict:
    """按域名选择解析函数"""
    netloc = urlparse(url).netloc

    if "people.com.cn" in netloc:
        data = parse_people_cn(html)
        source = "people"
    elif "globaltimes.cn" in netloc:
        data = parse_globaltimes_cn(html)
        source = "globaltimes_cn"
    elif "reuters.com" in netloc:
        data = parse_reuters(html)
        source = "reuters"
    else:
        # 其它网站统统走通用解析，source 记录原始域名方便后面分组分析
        data = parse_generic(html)
        source = netloc

    data["source"] = source
    data["url"] = url
    return data


# ------------------- 主抓取逻辑 -------------------


def crawl_from_url_file(path, country: str) -> pd.DataFrame:
    """
    从一个 URL 列表文件中逐条抓取。
    - path: 文本文件，每行一个 URL，可用 # 开头做注释
    - country: "CN" / "US" 等，用于后续分析打标签
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        urls = []
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            urls.append(line)
        # 去重但保持原有顺序
        urls = list(dict.fromkeys(urls))

    for url in tqdm(urls, desc=f"Crawling {country} news"):
        html = fetch_html(url)
        if not html:
            continue
        info = parse_article(url, html)
        info["country"] = country
        rows.append(info)

    if not rows:
        return pd.DataFrame(columns=["title", "date", "content", "source", "url", "country"])
    return pd.DataFrame(rows)


def main():
    cn_url_file = CONFIG_DIR / "cn_urls.txt"
    us_url_file = CONFIG_DIR / "us_urls.txt"

    print(f"[info] Loading CN urls from {cn_url_file}")
    cn_df = crawl_from_url_file(cn_url_file, country="CN")

    print(f"[info] Loading US urls from {us_url_file}")
    us_df = crawl_from_url_file(us_url_file, country="US")

    all_df = pd.concat([cn_df, us_df], ignore_index=True)
    out_path = RAW_DIR / "news_raw.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("Saved news to", out_path)
    print(all_df.head())


if __name__ == "__main__":
    main()
