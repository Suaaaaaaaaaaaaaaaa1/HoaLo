"""
Data Cleaning: Raw Apify output → posts_cleaned.csv
Matches the format expected by the NLP notebook.
Handles: column mapping, datetime extraction, engagement calculation,
media type detection, reaction estimates, deduplication.
"""
import os
import logging
import argparse
from pathlib import Path

import pandas as pd

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_raw_data(input_dir: Path) -> pd.DataFrame:
    csv_path = input_dir / "hoa_lo_posts.csv"
    json_path = input_dir / "hoa_lo_raw.json"

    if json_path.exists():
        logger.info(f"Loading raw JSON: {json_path}")
        df = pd.read_json(json_path, encoding="utf-8")
    elif csv_path.exists():
        logger.info(f"Loading raw CSV: {csv_path}")
        df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)
    else:
        logger.error(f"No data files found in {input_dir}")
        return pd.DataFrame()

    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"Columns: {list(df.columns)}")
    return df


def resolve_column(df: pd.DataFrame, candidates: list[str], default=None):
    for col in candidates:
        if col in df.columns:
            return df[col]
    return pd.Series(default, index=df.index) if default is not None else pd.Series(dtype="object", index=df.index)


def clean_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty:
        return df_raw

    df = pd.DataFrame()

    df["postUrl"] = resolve_column(df_raw, ["postUrl", "url", "postUrl", "link"])
    df["postId"] = resolve_column(df_raw, ["postId", "post_id", "id"])
    if df["postId"].isna().all() and df["postUrl"].notna().any():
        df["postId"] = df["postUrl"].str.extract(r"/posts/([^/?]+)")

    time_col = resolve_column(df_raw, ["time", "timestamp", "created_time", "date"])
    df["datetime"] = pd.to_datetime(time_col, utc=True, errors="coerce")
    df["datetime"] = df["datetime"].dt.tz_localize(None)
    df["date"] = df["datetime"].dt.date
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["year_month"] = df["datetime"].dt.to_period("M").astype(str)
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.day_name()

    df["text"] = resolve_column(df_raw, ["text", "message", "content"], default="").fillna("")
    df["text_length"] = df["text"].str.len()
    df["has_text"] = df["text_length"] > 0

    reactions_raw = resolve_column(df_raw, ["topReactionsCount", "reactions_total", "likes"])
    df["reactions_total"] = pd.to_numeric(reactions_raw, errors="coerce").fillna(0).astype(int)

    comments_raw = resolve_column(df_raw, ["comments", "comment_count"])
    if comments_raw.apply(lambda x: isinstance(x, dict)).any():
        comments_raw = comments_raw.apply(lambda x: x.get("count", 0) if isinstance(x, dict) else x)
    df["comment_count"] = pd.to_numeric(comments_raw, errors="coerce").fillna(0).astype(int)

    shares_raw = resolve_column(df_raw, ["shares", "share_count"])
    if shares_raw.apply(lambda x: isinstance(x, dict)).any():
        shares_raw = shares_raw.apply(lambda x: x.get("count", 0) if isinstance(x, dict) else x)
    df["share_count"] = pd.to_numeric(shares_raw, errors="coerce").fillna(0).astype(int)

    views_raw = resolve_column(df_raw, ["viewsCount", "views_count", "videoViewCount"])
    df["views_count"] = pd.to_numeric(views_raw, errors="coerce").fillna(0).astype(int)

    df["engagement_total"] = df["reactions_total"] + df["comment_count"] + df["share_count"]

    is_video = resolve_column(df_raw, ["isVideo", "is_video"])
    df["isVideo"] = is_video.fillna(False).astype(bool)

    media_url = pd.Series(dtype="object", index=df.index)
    for candidate in ["media/0/url", "media/0/image/url", "imageUrl", "image_url", "media_url"]:
        if candidate in df_raw.columns and df_raw[candidate].notna().sum() > 0:
            media_url = df_raw[candidate]
            break
    df["media_url"] = media_url

    df["media_type"] = "None"
    df.loc[df["isVideo"], "media_type"] = "Video"
    df.loc[(df["media_url"].notna()) & (~df["isVideo"]), "media_type"] = "Photo"
    df.loc[(df["media_type"] == "None") & (df["has_text"]) & (df["media_url"].isna()), "media_type"] = "Text-only"

    love_raw = resolve_column(df_raw, ["reactionLoveCount", "love_count"])
    df["love_count"] = pd.to_numeric(love_raw, errors="coerce").fillna(0).astype(int)
    wow_raw = resolve_column(df_raw, ["reactionWowCount", "wow_count"])
    df["wow_count"] = pd.to_numeric(wow_raw, errors="coerce").fillna(0).astype(int)

    df["estimated_like"] = (df["reactions_total"] * 0.65).astype(int)
    df["estimated_care"] = (df["reactions_total"] * 0.02).astype(int)
    df["estimated_haha"] = (df["reactions_total"] * 0.05).astype(int)
    df["estimated_sad"] = (df["reactions_total"] * 0.005).astype(int)
    df["estimated_angry"] = (df["reactions_total"] * 0.005).astype(int)

    before = len(df)
    df = df.drop_duplicates(subset=["postId"], keep="first")
    if before > len(df):
        logger.info(f"Removed {before - len(df)} duplicate posts")

    df = df.dropna(subset=["datetime"])
    df = df.sort_values("datetime", ascending=True).reset_index(drop=True)

    logger.info(f"Cleaned: {len(df)} posts")
    logger.info(f"  Date range: {df['date'].min()} → {df['date'].max()}")
    logger.info(f"  With text: {df['has_text'].sum()}")
    logger.info(f"  Total engagement: {df['engagement_total'].sum():,}")
    logger.info(f"  Media types: {df['media_type'].value_counts().to_dict()}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Clean raw Apify data → posts_cleaned.csv")
    parser.add_argument("--input", default="data/raw")
    parser.add_argument("--output", default="data/cleaned")
    parser.add_argument("--config", default="config/pipeline.yaml")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_raw_data(input_dir)
    if df_raw.empty:
        logger.error("No data to clean")
        return

    df_clean = clean_data(df_raw)

    output_path = output_dir / "posts_cleaned.csv"
    df_clean.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved to {output_path}")

    stats = {
        "total_posts": len(df_clean),
        "with_text": int(df_clean["has_text"].sum()),
        "date_min": str(df_clean["date"].min()),
        "date_max": str(df_clean["date"].max()),
        "engagement_total": int(df_clean["engagement_total"].sum()),
        "avg_engagement": round(float(df_clean["engagement_total"].mean()), 1),
        "media_types": df_clean["media_type"].value_counts().to_dict(),
    }
    import json
    with open(output_dir / "clean_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Stats saved")


if __name__ == "__main__":
    main()
