"""
Data Cleaning: 3 raw files → posts_cleaned.csv, comments_cleaned.csv, reviews_cleaned.csv
"""
import os
import logging
import argparse
from pathlib import Path

import pandas as pd

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def clean_posts(input_dir: Path) -> pd.DataFrame:
    candidates = ["post (+vid).csv", "post_(+vid).csv", "posts.csv", "hoa_lo_posts.csv", "hoa_lo_raw.json"]
    df_raw = None
    for name in candidates:
        p = input_dir / name
        if p.exists():
            df_raw = pd.read_json(p, encoding="utf-8") if name.endswith(".json") else pd.read_csv(p, encoding="utf-8", low_memory=False)
            logger.info(f"Loaded {name}: {len(df_raw)} rows, {len(df_raw.columns)} cols")
            break
    if df_raw is None:
        for f in sorted(input_dir.glob("*.csv")) + sorted(input_dir.glob("*.json")):
            if "comment" not in f.name.lower() and "review" not in f.name.lower():
                df_raw = pd.read_json(f) if f.suffix == ".json" else pd.read_csv(f, encoding="utf-8", low_memory=False)
                logger.info(f"Loaded fallback {f.name}: {len(df_raw)} rows")
                break
    if df_raw is None:
        logger.error("No post data found")
        return pd.DataFrame()

    df = pd.DataFrame()
    col_map = {
        "postUrl": ["postUrl", "url"],
        "timestamp": ["timestamp"],
        "time": ["time"],
        "text": ["text", "message", "content"],
        "isVideo": ["isVideo", "is_video"],
        "topReactionsCount": ["topReactionsCount", "likes", "reactions_total"],
        "comments": ["comments", "comment_count"],
        "shares": ["shares", "share_count"],
        "viewsCount": ["viewsCount", "views_count", "videoViewCount"],
        "reactionLoveCount": ["reactionLoveCount", "love_count"],
        "reactionWowCount": ["reactionWowCount", "wow_count"],
    }
    for target, sources in col_map.items():
        for s in sources:
            if s in df_raw.columns:
                df[target] = df_raw[s]
                break
        if target not in df.columns:
            df[target] = None

    time_src = df["time"] if df["time"].notna().any() else df["timestamp"]
    df["datetime"] = pd.to_datetime(time_src, utc=True, errors="coerce").dt.tz_localize(None)
    df["date"] = df["datetime"].dt.date
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["year_month"] = df["datetime"].dt.to_period("M").astype(str)
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.day_name()

    df["text"] = df["text"].fillna("")
    df["text_length"] = df["text"].str.len()
    df["has_text"] = df["text_length"] > 0

    for col in ["topReactionsCount", "comments", "shares", "viewsCount"]:
        vals = df[col]
        if vals.apply(lambda x: isinstance(x, dict)).any():
            vals = vals.apply(lambda x: x.get("count", x.get("total", 0)) if isinstance(x, dict) else x)
        df[col] = pd.to_numeric(vals, errors="coerce").fillna(0).astype(int)

    df["reactions_total"] = df["topReactionsCount"]
    df["comment_count"] = df["comments"]
    df["share_count"] = df["shares"]
    df["views_count"] = df["viewsCount"]
    df["engagement_total"] = df["reactions_total"] + df["comment_count"] + df["share_count"]

    media_url = pd.Series(dtype="object", index=df.index)
    for c in ["media/0/url", "media/0/image/url", "imageUrl", "image_url"]:
        if c in df_raw.columns and df_raw[c].notna().sum() > 0:
            media_url = df_raw[c]
            break
    df["media_url"] = media_url
    df["isVideo"] = df["isVideo"].fillna(False).astype(bool)
    df["media_type"] = "None"
    df.loc[df["isVideo"], "media_type"] = "Video"
    df.loc[(df["media_url"].notna()) & (~df["isVideo"]), "media_type"] = "Photo"
    df.loc[(df["media_type"] == "None") & df["has_text"] & df["media_url"].isna(), "media_type"] = "Text-only"

    for col in ["reactionLoveCount", "reactionWowCount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df["love_count"] = df["reactionLoveCount"]
    df["wow_count"] = df["reactionWowCount"]
    df["estimated_like"] = (df["reactions_total"] * 0.65).astype(int)
    df["estimated_care"] = (df["reactions_total"] * 0.02).astype(int)
    df["estimated_haha"] = (df["reactions_total"] * 0.05).astype(int)
    df["estimated_sad"] = (df["reactions_total"] * 0.005).astype(int)
    df["estimated_angry"] = (df["reactions_total"] * 0.005).astype(int)

    post_url = df.get("postUrl", pd.Series(dtype="str"))
    df["postId"] = post_url.str.extract(r"/posts/([^/?]+)") if post_url.notna().any() else df.index.astype(str)
    df["postId"] = df["postId"].fillna(pd.Series(df.index.astype(str), index=df.index))
    df = df.drop_duplicates(subset=["postId"], keep="first")
    df = df.dropna(subset=["datetime"])
    df = df.sort_values("datetime", ascending=True).reset_index(drop=True)

    keep = [
        "postId", "postUrl", "datetime", "date", "year", "month", "day", "year_month", "hour", "weekday",
        "text", "text_length", "has_text", "media_type", "media_url", "isVideo",
        "reactions_total", "comment_count", "share_count", "views_count", "engagement_total",
        "estimated_like", "love_count", "estimated_care", "estimated_haha", "wow_count", "estimated_sad", "estimated_angry",
    ]
    df = df[[c for c in keep if c in df.columns]]
    logger.info(f"Cleaned posts: {len(df)} rows, date {df['date'].min()} → {df['date'].max()}, engagement {df['engagement_total'].sum():,}")
    return df


def clean_comments(input_dir: Path) -> pd.DataFrame:
    for name in ["comment.csv", "comments.csv"]:
        p = input_dir / name
        if p.exists():
            df_raw = pd.read_csv(p, encoding="utf-8", low_memory=False)
            df = pd.DataFrame()
            for c in ["comments/0/text", "text", "comment_text"]:
                if c in df_raw.columns:
                    df["text"] = df_raw[c]
                    break
            for c in ["comments/0/date", "date", "comment_date"]:
                if c in df_raw.columns:
                    df["date"] = df_raw[c]
                    break
            logger.info(f"Cleaned comments: {len(df)} rows")
            return df
    logger.warning("No comment file found")
    return pd.DataFrame()


def clean_reviews(input_dir: Path) -> pd.DataFrame:
    for name in ["reviews.csv", "review.csv"]:
        p = input_dir / name
        if p.exists():
            df = pd.read_csv(p, encoding="utf-8")
            if "date" in df.columns:
                df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
            if "text" in df.columns:
                df["text"] = df["text"].fillna("")
                df["text_length"] = df["text"].str.len()
                pos_words = ["good", "great", "excellent", "amazing", "wonderful", "love", "nice"]
                neg_words = ["bad", "poor", "terrible", "disappointed", "waste"]
                df["sentiment_score"] = df["text"].str.lower().apply(
                    lambda x: sum(w in str(x) for w in pos_words) - sum(w in str(x) for w in neg_words))
                df["sentiment"] = "Neutral"
                df.loc[df["sentiment_score"] > 0, "sentiment"] = "Positive"
                df.loc[df["sentiment_score"] < 0, "sentiment"] = "Negative"
            logger.info(f"Cleaned reviews: {len(df)} rows")
            return df
    logger.warning("No review file found")
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw")
    parser.add_argument("--output", default="data/cleaned")
    parser.add_argument("--config", default="config/pipeline.yaml")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    inp = Path(args.input)

    df_posts = clean_posts(inp)
    if not df_posts.empty:
        df_posts.to_csv(out / "posts_cleaned.csv", index=False, encoding="utf-8-sig")

    df_comments = clean_comments(inp)
    if not df_comments.empty:
        df_comments.to_csv(out / "comments_cleaned.csv", index=False, encoding="utf-8-sig")

    df_reviews = clean_reviews(inp)
    if not df_reviews.empty:
        df_reviews.to_csv(out / "reviews_cleaned.csv", index=False, encoding="utf-8-sig")

    logger.info("Data cleaning complete")


if __name__ == "__main__":
    main()
