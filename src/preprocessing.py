"""
Vietnamese text preprocessing pipeline.
Handles tokenization (underthesea), stopword removal, URL/emoji cleaning.
"""
import re
import os
import logging
import argparse
from pathlib import Path

import pandas as pd
import yaml
from underthesea import word_tokenize

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
SPECIAL_CHARS = re.compile(r"[^\w\s]", re.UNICODE)
MULTI_SPACES = re.compile(r"\s+")


def load_stopwords(path: str) -> set[str]:
    stopwords = set()
    p = Path(path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word and not word.startswith("#"):
                    stopwords.add(word.lower())
        logger.info(f"Loaded {len(stopwords)} stopwords from {path}")
    else:
        logger.warning(f"Stopwords file not found: {path}. Proceeding without stopwords.")
    return stopwords


def clean_text(text: str, config: dict) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""

    if config.get("remove_urls", True):
        text = URL_PATTERN.sub(" ", text)

    if config.get("remove_emojis", True):
        text = EMOJI_PATTERN.sub(" ", text)

    text = SPECIAL_CHARS.sub(" ", text)

    if config.get("lowercase", True):
        text = text.lower()

    text = MULTI_SPACES.sub(" ", text).strip()
    return text


def tokenize_vietnamese(text: str, stopwords: set, config: dict) -> str:
    if not text:
        return ""

    tokens = word_tokenize(text, format="text").split()

    min_len = config.get("min_token_length", 2)
    max_len = config.get("max_token_length", 20)

    filtered = [
        t for t in tokens
        if t.lower() not in stopwords
        and min_len <= len(t) <= max_len
        and not t.isdigit()
    ]

    return " ".join(filtered)


def process_dataframe(df: pd.DataFrame, stopwords: set, config: dict) -> pd.DataFrame:
    result = df.copy()

    text_col = "message" if "message" in result.columns else "text"
    result["text_cleaned"] = result[text_col].apply(lambda x: clean_text(x, config))
    result["text_tokenized"] = result["text_cleaned"].apply(
        lambda x: tokenize_vietnamese(x, stopwords, config)
    )
    result["token_count"] = result["text_tokenized"].apply(lambda x: len(x.split()) if x else 0)

    before = len(result)
    result = result[result["token_count"] > 0].reset_index(drop=True)
    logger.info(f"Dropped {before - len(result)} empty rows after preprocessing")

    return result


def main():
    parser = argparse.ArgumentParser(description="Preprocess Vietnamese Facebook posts")
    parser.add_argument("--input", default="data/raw")
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--config", default="config/pipeline.yaml")
    parser.add_argument("--stopwords", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        full_config = yaml.safe_load(f)
    config = full_config["preprocessing"]

    stopwords_path = args.stopwords or config.get("stopwords_path", "data/vietnamese_stopwords.txt")
    stopwords = load_stopwords(stopwords_path)

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(input_dir.glob("*_posts.csv")) + list(input_dir.glob("*_cleaned.csv"))
    if not csv_files:
        logger.error(f"No CSV files found in {input_dir}")
        return

    seen = set()
    for csv_path in csv_files:
        if csv_path.name in seen:
            continue
        seen.add(csv_path.name)
        logger.info(f"=== Processing: {csv_path.name} ===")
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows")

        processed = process_dataframe(df, stopwords, config)
        stem = csv_path.stem.replace("_posts", "").replace("_cleaned", "").replace("posts_", "")
        if not stem:
            stem = "hoa_lo"
        out_path = output_dir / f"{stem}_processed.csv"
        processed.to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.info(f"Saved {len(processed)} processed rows to {out_path}")


if __name__ == "__main__":
    main()
