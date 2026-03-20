"""
Facebook Page Posts Scraper via Apify.
Uses apify/facebook-posts-scraper Actor to collect posts from Hỏa Lò page.
"""
import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import yaml
from apify_client import ApifyClient
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ACTOR_ID = "apify/facebook-posts-scraper"


def load_config(path: str = "config/pipeline.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def scrape_page(client: ApifyClient, page_url: str, config: dict) -> list[dict]:
    scraper_cfg = config.get("scraper", {})
    max_posts = scraper_cfg.get("max_posts", 500)

    run_input = {
        "startUrls": [{"url": page_url}],
        "resultsLimit": max_posts,
    }

    logger.info(f"Starting Apify Actor: {ACTOR_ID}")
    logger.info(f"Page: {page_url} | Max posts: {max_posts}")

    run = client.actor(ACTOR_ID).call(run_input=run_input)

    dataset_id = run["defaultDatasetId"]
    logger.info(f"Run finished. Dataset: https://console.apify.com/storage/datasets/{dataset_id}")

    items = list(client.dataset(dataset_id).iterate_items())
    logger.info(f"Collected {len(items)} raw items")
    return items


def normalize_items(items: list[dict]) -> pd.DataFrame:
    rows = []
    for item in items:
        likes = item.get("likes", 0) or 0
        comments = item.get("comments", 0) or 0
        shares = item.get("shares", 0) or 0

        if isinstance(likes, dict):
            likes = likes.get("total", 0)
        if isinstance(comments, dict):
            comments = comments.get("count", 0)

        row = {
            "post_id": item.get("postId", item.get("id", "")),
            "message": item.get("text", item.get("message", "")),
            "created_time": item.get("time", item.get("timestamp", "")),
            "type": item.get("type", ""),
            "likes": likes,
            "comments": comments,
            "shares": shares,
            "url": item.get("url", item.get("postUrl", "")),
            "page_name": item.get("pageName", ""),
            "image_url": item.get("imageUrl", ""),
            "video_url": item.get("videoUrl", ""),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if "created_time" in df.columns and not df.empty:
        df["created_time"] = pd.to_datetime(df["created_time"], errors="coerce")
        df = df.sort_values("created_time", ascending=False)

    return df


def main():
    parser = argparse.ArgumentParser(description="Scrape Facebook page posts via Apify")
    parser.add_argument("--config", default="config/pipeline.yaml")
    parser.add_argument("--output", default="data/raw")
    args = parser.parse_args()

    token = os.getenv("APIFY_API_TOKEN")
    if not token:
        logger.error("APIFY_API_TOKEN not set. Add it to .env file.")
        sys.exit(1)

    config = load_config(args.config)
    client = ApifyClient(token)

    page_url = config.get("page", {}).get("url", "https://www.facebook.com/hoaloprisonrelic/")
    page_name = config.get("page", {}).get("name", "hoa_lo")

    logger.info(f"=== Scraping: {page_name} ({page_url}) ===")
    items = scrape_page(client, page_url, config)

    if not items:
        logger.warning("No posts collected. Check Apify token and page URL.")
        sys.exit(1)

    df = normalize_items(items)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{page_name}_posts.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved {len(df)} posts to {csv_path}")

    json_path = output_dir / f"{page_name}_raw.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Saved raw JSON to {json_path}")

    meta = {
        "scraped_at": datetime.now().isoformat(),
        "actor": ACTOR_ID,
        "page_url": page_url,
        "total_posts": len(df),
        "date_range": {
            "earliest": str(df["created_time"].min()) if not df.empty else None,
            "latest": str(df["created_time"].max()) if not df.empty else None,
        },
    }
    with open(output_dir / "scrape_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info("Metadata saved")


if __name__ == "__main__":
    main()
