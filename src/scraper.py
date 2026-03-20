"""
Apify Data Fetcher.
Mode 1 (default): Fetch data from the latest Apify dataset (pre-scheduled run) — instant.
Mode 2 (--trigger): Trigger a new Actor run and wait — slow, 10-15min.
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


def fetch_latest_dataset(client: ApifyClient, config: dict) -> list[dict]:
    scraper_cfg = config.get("scraper", {})
    dataset_id = scraper_cfg.get("dataset_id")

    if dataset_id:
        logger.info(f"Fetching from specified dataset: {dataset_id}")
        items = list(client.dataset(dataset_id).iterate_items())
    else:
        logger.info(f"Fetching latest run dataset from Actor: {ACTOR_ID}")
        runs = client.actor(ACTOR_ID).runs().list(limit=1, desc=True)
        if not runs.items:
            logger.error("No previous runs found. Run the Actor manually on Apify Console first, or use --trigger.")
            return []
        last_run = runs.items[0]
        dataset_id = last_run.get("defaultDatasetId")
        run_status = last_run.get("status")
        finished_at = last_run.get("finishedAt", "unknown")
        logger.info(f"Latest run: status={run_status}, finished={finished_at}, dataset={dataset_id}")

        if run_status != "SUCCEEDED":
            logger.warning(f"Latest run status is '{run_status}', data may be incomplete")

        items = list(client.dataset(dataset_id).iterate_items())

    logger.info(f"Fetched {len(items)} items from dataset")
    return items


def trigger_new_run(client: ApifyClient, config: dict) -> list[dict]:
    scraper_cfg = config.get("scraper", {})
    page_url = config.get("page", {}).get("url", "https://www.facebook.com/hoaloprisonrelic/")
    max_posts = scraper_cfg.get("max_posts", 500)

    run_input = {
        "startUrls": [{"url": page_url}],
        "resultsLimit": max_posts,
    }

    logger.info(f"Triggering new Actor run: {ACTOR_ID}")
    logger.info(f"Page: {page_url} | Max posts: {max_posts}")
    logger.info("This will take 10-15 minutes...")

    run = client.actor(ACTOR_ID).call(run_input=run_input)
    dataset_id = run["defaultDatasetId"]
    logger.info(f"Run finished. Dataset: https://console.apify.com/storage/datasets/{dataset_id}")

    items = list(client.dataset(dataset_id).iterate_items())
    logger.info(f"Collected {len(items)} items")
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
    parser = argparse.ArgumentParser(description="Fetch Facebook data from Apify")
    parser.add_argument("--config", default="config/pipeline.yaml")
    parser.add_argument("--output", default="data/raw")
    parser.add_argument("--trigger", action="store_true", help="Trigger new Actor run instead of fetching latest dataset")
    args = parser.parse_args()

    token = os.getenv("APIFY_API_TOKEN")
    if not token:
        logger.error("APIFY_API_TOKEN not set. Add it to .env file.")
        sys.exit(1)

    config = load_config(args.config)
    client = ApifyClient(token)
    page_name = config.get("page", {}).get("name", "hoa_lo")

    if args.trigger:
        items = trigger_new_run(client, config)
    else:
        items = fetch_latest_dataset(client, config)

    if not items:
        logger.error("No data fetched. Check Apify Console for available datasets.")
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
        "fetched_at": datetime.now().isoformat(),
        "mode": "trigger" if args.trigger else "fetch_latest",
        "actor": ACTOR_ID,
        "total_posts": len(df),
        "date_range": {
            "earliest": str(df["created_time"].min()) if not df.empty else None,
            "latest": str(df["created_time"].max()) if not df.empty else None,
        },
    }
    with open(output_dir / "scrape_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info("Done!")


if __name__ == "__main__":
    main()
