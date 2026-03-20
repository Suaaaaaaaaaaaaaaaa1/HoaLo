"""
Sentiment analysis for Vietnamese Facebook posts.
Uses underthesea sentiment classifier with rule-based post-processing.
"""
import os
import logging
import argparse
from pathlib import Path

import pandas as pd
import yaml
from underthesea import sentiment

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

POSITIVE_KEYWORDS = {
    "tuyệt vời", "đẹp", "hay", "thích", "yêu", "xuất sắc", "tốt", "tuyệt",
    "ấn tượng", "cảm ơn", "cảm_ơn", "trải nghiệm", "tham quan", "đáng",
    "giá trị", "lịch sử", "ý nghĩa", "nên đến", "nên_đến", "recommend",
    "amazing", "beautiful", "great", "love", "wonderful",
}

NEGATIVE_KEYWORDS = {
    "tệ", "chán", "thất vọng", "thất_vọng", "xấu", "dở", "kém",
    "đông", "nóng", "mệt", "không thích", "không_thích", "phí tiền",
    "lãng phí", "lãng_phí", "boring", "bad", "terrible", "waste",
}


def classify_sentiment(text: str, method: str = "underthesea") -> str:
    if not isinstance(text, str) or not text.strip():
        return "neutral"

    if method == "underthesea":
        try:
            result = sentiment(text)
            if isinstance(result, str):
                return result.lower()
            return "neutral"
        except Exception:
            return "neutral"

    return "neutral"


def apply_post_processing(row: pd.Series) -> str:
    text = str(row.get("text_tokenized", "")).lower()
    base_label = row.get("sentiment_raw", "neutral")

    pos_count = sum(1 for kw in POSITIVE_KEYWORDS if kw in text)
    neg_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text)

    if base_label == "neutral":
        if pos_count >= 2 and neg_count == 0:
            return "positive"
        if neg_count >= 2 and pos_count == 0:
            return "negative"

    if base_label == "negative" and neg_count == 0 and pos_count >= 2:
        return "positive"
    if base_label == "positive" and pos_count == 0 and neg_count >= 2:
        return "negative"

    return base_label


def analyze_sentiment(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    nlp_config = config.get("nlp", {}).get("sentiment", {})
    method = nlp_config.get("method", "underthesea")
    use_rules = nlp_config.get("post_process_rules", True)

    result = df.copy()

    logger.info(f"Running sentiment classification (method={method})...")
    result["sentiment_raw"] = result["text_tokenized"].apply(lambda x: classify_sentiment(x, method))

    if use_rules:
        logger.info("Applying post-processing rules...")
        result["sentiment"] = result.apply(apply_post_processing, axis=1)
    else:
        result["sentiment"] = result["sentiment_raw"]

    dist = result["sentiment"].value_counts(normalize=True) * 100
    logger.info("=== Sentiment Distribution ===")
    for label in ["neutral", "positive", "negative"]:
        logger.info(f"  {label}: {dist.get(label, 0):.1f}%")

    return result


def main():
    parser = argparse.ArgumentParser(description="Run sentiment analysis on processed posts")
    parser.add_argument("--input", default="data/processed")
    parser.add_argument("--output", default="data/results/sentiment")
    parser.add_argument("--config", default="config/pipeline.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(input_dir.glob("*_processed.csv"))
    if not csv_files:
        logger.error(f"No processed CSV files found in {input_dir}")
        return

    for csv_path in csv_files:
        logger.info(f"=== Analyzing: {csv_path.name} ===")
        df = pd.read_csv(csv_path)

        result = analyze_sentiment(df, config)

        out_name = csv_path.name.replace("_processed.csv", "_sentiment.csv")
        result.to_csv(output_dir / out_name, index=False, encoding="utf-8-sig")
        logger.info(f"Saved to {output_dir / out_name}")

        summary = result["sentiment"].value_counts().to_frame("count")
        summary["percent"] = (summary["count"] / summary["count"].sum() * 100).round(1)
        summary_path = output_dir / csv_path.name.replace("_processed.csv", "_sentiment_summary.csv")
        summary.to_csv(summary_path, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
