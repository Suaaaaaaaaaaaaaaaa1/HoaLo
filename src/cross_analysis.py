"""
Cross Analysis: Topic × Sentiment × Engagement.
Merges NLP results, generates insights, enriched dataset, and comprehensive visualizations.
"""
import os
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SENT_COLORS = {"positive": "#2ecc71", "neutral": "#3498db", "negative": "#e74c3c"}


def load_and_merge(sentiment_dir: Path, topics_dir: Path, cleaned_dir: Path) -> pd.DataFrame:
    sent_files = list(sentiment_dir.glob("*_sentiment.csv"))
    topics_files = list(topics_dir.glob("*_doc_topics.csv"))
    cleaned_path = cleaned_dir / "posts_cleaned.csv"

    if not sent_files:
        logger.error(f"No sentiment CSV found in {sentiment_dir}")
        logger.error(f"  Contents: {list(sentiment_dir.glob('*')) if sentiment_dir.exists() else 'dir not found'}")
        return pd.DataFrame()

    sent_path = sent_files[0]
    logger.info(f"Loading sentiment: {sent_path.name}")
    df_sent = pd.read_csv(sent_path, encoding="utf-8")

    topic_cols = []
    if topics_files:
        topics_path = topics_files[0]
        logger.info(f"Loading topics: {topics_path.name}")
        df_topics = pd.read_csv(topics_path, encoding="utf-8")
        topic_weight_cols = [c for c in df_topics.columns if c.startswith("topic_") and c.endswith("_weight")]
        merge_cols = ["post_id", "dominant_topic"] + topic_weight_cols
        available = [c for c in merge_cols if c in df_topics.columns]

        if "post_id" in df_topics.columns and "post_id" in df_sent.columns:
            df = df_sent.merge(df_topics[available], on="post_id", how="left")
        elif len(df_sent) == len(df_topics):
            logger.info("No post_id match, merging by index (same row count)")
            for col in available:
                if col != "post_id":
                    df_sent[col] = df_topics[col].values
            df = df_sent.copy()
        else:
            logger.warning("Cannot merge topics — no post_id match and different row counts")
            df = df_sent.copy()

        topic_cols = [c for c in topic_weight_cols if c in df.columns]
    else:
        logger.warning(f"No doc_topics CSV found in {topics_dir}, using sentiment only")
        df = df_sent.copy()

    if "dominant_topic" not in df.columns and topic_cols:
        df["dominant_topic"] = df[topic_cols].idxmax(axis=1)
    elif "dominant_topic" not in df.columns and not topic_cols:
        logger.warning("No topic weight columns available for dominant_topic")

    if cleaned_path.exists():
        df_clean = pd.read_csv(cleaned_path, encoding="utf-8")
        eng_cols = ["postId", "engagement_total", "reactions_total", "comment_count", "share_count", "views_count", "media_type", "year_month", "hour", "weekday"]
        available_eng = [c for c in eng_cols if c in df_clean.columns]
        if "postId" in df_clean.columns and "post_id" in df.columns:
            df = df.merge(df_clean[available_eng], left_on="post_id", right_on="postId", how="left")

    if "engagement_total" not in df.columns:
        for likes_col in ["likes", "reactions_total"]:
            if likes_col in df.columns:
                df["engagement_total"] = pd.to_numeric(df.get(likes_col, 0), errors="coerce").fillna(0) + \
                                          pd.to_numeric(df.get("comments", df.get("comment_count", 0)), errors="coerce").fillna(0) + \
                                          pd.to_numeric(df.get("shares", df.get("share_count", 0)), errors="coerce").fillna(0)
                break

    if "topic_id" not in df.columns and "dominant_topic" in df.columns:
        df["topic_id"] = df["dominant_topic"].str.extract(r"(\d+)").astype(float).astype("Int64")

    logger.info(f"Merged dataset: {len(df)} rows, {len(df.columns)} cols")
    return df


def load_topic_labels(topics_dir: Path) -> dict:
    json_files = list(topics_dir.glob("*_topics.json"))
    if not json_files:
        return {}
    with open(json_files[0]) as f:
        data = json.load(f)
    labels = {}
    for t in data.get("topics", []):
        tid = t["topic_id"]
        words = ", ".join(w for w, _ in t["top_words"][:5])
        labels[tid] = t.get("label", f"Topic_{tid}") + f" ({words})"
    return labels


def analyze_topic_sentiment(df: pd.DataFrame, topic_labels: dict) -> dict:
    if "topic_id" not in df.columns or "sentiment" not in df.columns:
        return {}

    df_valid = df[df["topic_id"].notna() & (df["topic_id"] >= 0)].copy()
    if df_valid.empty:
        return {}

    cross = pd.crosstab(df_valid["topic_id"], df_valid["sentiment"], normalize="index") * 100
    logger.info("=== Topic × Sentiment (%) ===")
    logger.info(f"\n{cross.round(1)}")

    results = {}
    for tid in cross.index:
        label = topic_labels.get(int(tid), f"Topic_{int(tid)}")
        results[int(tid)] = {
            "label": label,
            "sentiment_pct": {col: round(float(cross.loc[tid, col]), 1) for col in cross.columns},
            "post_count": int((df_valid["topic_id"] == tid).sum()),
        }

    return results


def analyze_topic_engagement(df: pd.DataFrame, topic_labels: dict) -> dict:
    if "topic_id" not in df.columns or "engagement_total" not in df.columns:
        return {}

    df_valid = df[df["topic_id"].notna() & (df["topic_id"] >= 0)].copy()
    if df_valid.empty:
        return {}

    stats = df_valid.groupby("topic_id").agg(
        avg_engagement=("engagement_total", "mean"),
        median_engagement=("engagement_total", "median"),
        max_engagement=("engagement_total", "max"),
        avg_reactions=("reactions_total", "mean") if "reactions_total" in df_valid.columns else ("engagement_total", "mean"),
        avg_comments=("comment_count", "mean") if "comment_count" in df_valid.columns else ("engagement_total", "mean"),
        avg_shares=("share_count", "mean") if "share_count" in df_valid.columns else ("engagement_total", "mean"),
        post_count=("topic_id", "count"),
    ).round(1)

    logger.info("=== Engagement by Topic ===")
    logger.info(f"\n{stats}")

    best_tid = stats["avg_engagement"].idxmax()
    logger.info(f"Best topic: {int(best_tid)} — {topic_labels.get(int(best_tid), 'N/A')}")

    return {
        "stats": stats.to_dict(orient="index"),
        "best_topic_id": int(best_tid),
        "best_topic_label": topic_labels.get(int(best_tid), f"Topic_{int(best_tid)}"),
        "best_topic_avg_engagement": float(stats.loc[best_tid, "avg_engagement"]),
    }


def analyze_sentiment_engagement(df: pd.DataFrame) -> dict:
    if "sentiment" not in df.columns or "engagement_total" not in df.columns:
        return {}

    sent_eng = df.groupby("sentiment")["engagement_total"].agg(["mean", "median", "count"]).round(1)
    logger.info("=== Engagement by Sentiment ===")
    logger.info(f"\n{sent_eng}")

    best_sent = sent_eng["mean"].idxmax()
    text_corr = 0.0
    if "text_length" in df.columns:
        text_corr = float(df[["text_length", "engagement_total"]].corr().iloc[0, 1])

    return {
        "by_sentiment": sent_eng.to_dict(orient="index"),
        "best_sentiment": best_sent,
        "best_sentiment_avg": float(sent_eng.loc[best_sent, "mean"]),
        "text_length_corr": round(text_corr, 3),
    }


def plot_comprehensive(df: pd.DataFrame, topic_labels: dict, output_dir: Path):
    df_valid = df[df.get("topic_id", pd.Series(dtype=float)).notna()].copy()
    if df_valid.empty or "topic_id" not in df_valid.columns:
        logger.warning("No valid data for comprehensive plot")
        return

    df_plot = df_valid[df_valid["topic_id"] >= 0].copy()

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])
    cross_count = pd.crosstab(df_plot["topic_id"], df_plot["sentiment"])
    sns.heatmap(cross_count, annot=True, fmt="d", cmap="YlGnBu", ax=ax1)
    ax1.set_title("Topic × Sentiment Distribution", fontweight="bold", fontsize=14)
    ax1.set_xlabel("Sentiment")
    ax1.set_ylabel("Topic ID")

    if "engagement_total" in df_plot.columns:
        ax2 = fig.add_subplot(gs[1, 0])
        topic_eng = df_plot.groupby("topic_id")["engagement_total"].mean().sort_values(ascending=False)
        ax2.barh(range(len(topic_eng)), topic_eng.values, color="green", alpha=0.7)
        ax2.set_yticks(range(len(topic_eng)))
        ax2.set_yticklabels([f"Topic {int(i)}" for i in topic_eng.index])
        ax2.set_xlabel("Avg Engagement")
        ax2.set_title("Avg Engagement by Topic", fontweight="bold", fontsize=12)
        ax2.grid(axis="x", alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    topic_counts = df_plot["topic_id"].value_counts().sort_index()
    ax3.bar(range(len(topic_counts)), topic_counts.values, color="steelblue", alpha=0.7)
    ax3.set_xticks(range(len(topic_counts)))
    ax3.set_xticklabels([f"T{int(i)}" for i in topic_counts.index])
    ax3.set_ylabel("Number of Posts")
    ax3.set_title("Posts Distribution by Topic", fontweight="bold", fontsize=12)
    ax3.grid(axis="y", alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 2])
    cross_pct = pd.crosstab(df_plot["topic_id"], df_plot["sentiment"], normalize="index") * 100
    colors = [SENT_COLORS.get(c, "gray") for c in cross_pct.columns]
    cross_pct.plot(kind="bar", stacked=True, ax=ax4, color=colors)
    ax4.set_xlabel("Topic ID")
    ax4.set_ylabel("Percentage (%)")
    ax4.set_title("Sentiment % by Topic", fontweight="bold", fontsize=12)
    ax4.legend(title="Sentiment", loc="upper right")
    ax4.grid(axis="y", alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=0)

    if "engagement_total" in df_plot.columns:
        ax5 = fig.add_subplot(gs[2, 0])
        topic_eng_sorted = df_plot.groupby("topic_id")["engagement_total"].mean().sort_values(ascending=True)
        color_ramp = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(topic_eng_sorted)))
        ax5.barh(range(len(topic_eng_sorted)), topic_eng_sorted.values, color=color_ramp)
        ax5.set_yticks(range(len(topic_eng_sorted)))
        ax5.set_yticklabels([f"Topic {int(i)}" for i in topic_eng_sorted.index])
        ax5.set_xlabel("Avg Engagement")
        ax5.set_title("Topic Performance Ranking", fontweight="bold", fontsize=12)
        ax5.grid(axis="x", alpha=0.3)

    if "text_length" in df_plot.columns and "engagement_total" in df_plot.columns:
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.scatter(df_plot["text_length"], df_plot["engagement_total"], alpha=0.3, s=20)
        ax6.set_xlabel("Text Length (chars)")
        ax6.set_ylabel("Engagement")
        ax6.set_title("Engagement vs Text Length", fontweight="bold", fontsize=12)
        ax6.grid(alpha=0.3)
        corr = df_plot[["text_length", "engagement_total"]].corr().iloc[0, 1]
        ax6.text(0.05, 0.95, f"Corr: {corr:.3f}", transform=ax6.transAxes,
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    if "engagement_total" in df_plot.columns:
        ax7 = fig.add_subplot(gs[2, 2])
        sent_eng = df_plot.groupby("sentiment")["engagement_total"].mean()
        colors = [SENT_COLORS.get(s, "gray") for s in sent_eng.index]
        ax7.bar(range(len(sent_eng)), sent_eng.values, color=colors)
        ax7.set_xticks(range(len(sent_eng)))
        ax7.set_xticklabels(sent_eng.index)
        ax7.set_ylabel("Avg Engagement")
        ax7.set_title("Avg Engagement by Sentiment", fontweight="bold", fontsize=12)
        ax7.grid(axis="y", alpha=0.3)

    plt.suptitle("COMPREHENSIVE NLP INSIGHTS — Hỏa Lò", fontsize=18, fontweight="bold")
    plt.savefig(output_dir / "cross_analysis_comprehensive.png", dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved cross_analysis_comprehensive.png")


def export_enriched(df: pd.DataFrame, output_dir: Path):
    export_cols = [
        "post_id", "message", "text_tokenized", "created_time",
        "topic_id", "dominant_topic", "sentiment", "sentiment_raw",
        "engagement_total", "reactions_total", "comment_count", "share_count",
        "media_type", "year_month", "text_length",
    ]
    available = [c for c in export_cols if c in df.columns]
    df_export = df[available].copy()
    out_path = output_dir / "posts_nlp_enriched.csv"
    df_export.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved enriched dataset: {out_path} ({len(df_export)} rows)")


def generate_report(topic_sentiment: dict, topic_engagement: dict, sentiment_engagement: dict, output_dir: Path):
    report = {
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "topic_sentiment": topic_sentiment,
        "topic_engagement": topic_engagement,
        "sentiment_engagement": sentiment_engagement,
    }

    out_path = output_dir / "cross_analysis_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Saved report: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Cross-analysis: Topic × Sentiment × Engagement")
    parser.add_argument("--sentiment", default="data/results/sentiment")
    parser.add_argument("--topics", default="data/results/topics")
    parser.add_argument("--cleaned", default="data/cleaned")
    parser.add_argument("--output", default="data/results/cross_analysis")
    parser.add_argument("--figures", default="reports/figures")
    parser.add_argument("--config", default="config/pipeline.yaml")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(args.figures)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_merge(Path(args.sentiment), Path(args.topics), Path(args.cleaned))
    if df.empty:
        logger.error("No data to analyze")
        return

    topic_labels = load_topic_labels(Path(args.topics))

    topic_sentiment = analyze_topic_sentiment(df, topic_labels)
    topic_engagement = analyze_topic_engagement(df, topic_labels)
    sentiment_engagement = analyze_sentiment_engagement(df)

    plot_comprehensive(df, topic_labels, figures_dir)
    export_enriched(df, output_dir)
    generate_report(topic_sentiment, topic_engagement, sentiment_engagement, output_dir)

    logger.info("Cross-analysis complete!")


if __name__ == "__main__":
    main()
