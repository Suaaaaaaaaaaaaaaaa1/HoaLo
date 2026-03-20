"""
Generate all report visualizations: sentiment charts, topic word clouds, engagement comparisons.
"""
import os
import json
import logging
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PALETTE = {"positive": "#2ecc71", "neutral": "#3498db", "negative": "#e74c3c"}
PAGE_LABELS = {"hoa_lo": "Hỏa Lò", "co_do_hue": "Cố Đô Huế", "dinh_doc_lap": "Đình Độc Lập"}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


def plot_sentiment_distribution(sentiment_dir: Path, output_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Sentiment Distribution by Page", fontsize=16, fontweight="bold", y=1.02)

    csv_files = sorted(sentiment_dir.glob("*_sentiment.csv"))
    for i, csv_path in enumerate(csv_files[:3]):
        page_name = csv_path.stem.replace("_sentiment", "")
        df = pd.read_csv(csv_path)
        dist = df["sentiment"].value_counts()

        colors = [PALETTE.get(label, "#95a5a6") for label in dist.index]
        axes[i].pie(
            dist.values,
            labels=[f"{l}\n({v})" for l, v in zip(dist.index, dist.values)],
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 10},
        )
        axes[i].set_title(PAGE_LABELS.get(page_name, page_name))

    plt.tight_layout()
    plt.savefig(output_dir / "sentiment_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved sentiment_distribution.png")


def plot_sentiment_comparison(sentiment_dir: Path, output_dir: Path):
    records = []
    for csv_path in sorted(sentiment_dir.glob("*_sentiment.csv")):
        page_name = csv_path.stem.replace("_sentiment", "")
        df = pd.read_csv(csv_path)
        dist = df["sentiment"].value_counts(normalize=True) * 100
        for label in ["positive", "neutral", "negative"]:
            records.append({"page": PAGE_LABELS.get(page_name, page_name), "sentiment": label, "percent": dist.get(label, 0)})

    if not records:
        return

    comp_df = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=comp_df, x="page", y="percent", hue="sentiment", palette=PALETTE, ax=ax)
    ax.set_title("Sentiment Comparison Across Heritage Sites", fontweight="bold")
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("")
    ax.legend(title="Sentiment")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", fontsize=9, padding=2)

    plt.tight_layout()
    plt.savefig(output_dir / "sentiment_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved sentiment_comparison.png")


def plot_engagement_timeline(raw_dir: Path, output_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 5))

    for csv_path in sorted(raw_dir.glob("*_posts.csv")):
        page_name = csv_path.stem.replace("_posts", "")
        df = pd.read_csv(csv_path, parse_dates=["created_time"])
        df["engagement"] = df["likes"] + df["comments"] + df["shares"]
        monthly = df.set_index("created_time").resample("M")["engagement"].mean()
        ax.plot(monthly.index, monthly.values, marker="o", markersize=4, label=PAGE_LABELS.get(page_name, page_name))

    ax.set_title("Average Monthly Engagement", fontweight="bold")
    ax.set_ylabel("Avg Engagement (likes + comments + shares)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "engagement_timeline.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved engagement_timeline.png")


def plot_top_tfidf(topics_dir: Path, output_dir: Path):
    for json_path in sorted(topics_dir.glob("*_topics.json")):
        with open(json_path) as f:
            data = json.load(f)

        page_name = data["page"]
        terms = data.get("top_tfidf_terms", [])[:20]
        if not terms:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        terms_sorted = sorted(terms, key=lambda x: x["mean_tfidf"])
        ax.barh(
            [t["term"] for t in terms_sorted],
            [t["mean_tfidf"] for t in terms_sorted],
            color="#3498db",
        )
        ax.set_title(f"Top TF-IDF Terms — {PAGE_LABELS.get(page_name, page_name)}", fontweight="bold")
        ax.set_xlabel("Mean TF-IDF Score")
        plt.tight_layout()
        plt.savefig(output_dir / f"tfidf_top_{page_name}.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved tfidf_top_{page_name}.png")


def main():
    parser = argparse.ArgumentParser(description="Generate report visualizations")
    parser.add_argument("--sentiment", default="data/results/sentiment")
    parser.add_argument("--topics", default="data/results/topics")
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--output", default="reports/figures")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    sentiment_dir = Path(args.sentiment)
    topics_dir = Path(args.topics)
    raw_dir = Path(args.raw)

    if sentiment_dir.exists():
        plot_sentiment_distribution(sentiment_dir, output_dir)
        plot_sentiment_comparison(sentiment_dir, output_dir)

    if raw_dir.exists():
        plot_engagement_timeline(raw_dir, output_dir)

    if topics_dir.exists():
        plot_top_tfidf(topics_dir, output_dir)

    logger.info(f"All figures saved to {output_dir}")


if __name__ == "__main__":
    main()
