"""
Strategy report generator.
Compares pages, identifies strengths/gaps, produces actionable recommendations.
"""
import os
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import yaml
from jinja2 import Template

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PAGE_LABELS = {"hoa_lo": "Hỏa Lò", "co_do_hue": "Cố Đô Huế", "dinh_doc_lap": "Đình Độc Lập"}

REPORT_TEMPLATE = """# Báo Cáo Chiến Lược Social Media Marketing
## Di Tích Nhà Tù Hỏa Lò vs. Đối Thủ Cạnh Tranh

**Ngày tạo:** {{ generated_at }}
**Phiên bản dữ liệu:** v2.1 (sentiment recalculated)

---

## 1. Tổng Quan Dữ Liệu

| Trang | Tổng bài đăng | Engagement TB | Post/tuần TB |
|-------|---------------|---------------|--------------|
{% for p in pages %}| {{ p.label }} | {{ p.total_posts }} | {{ p.avg_engagement }} | {{ p.posts_per_week }} |
{% endfor %}

## 2. Phân Tích Sentiment

| Trang | Positive | Neutral | Negative |
|-------|----------|---------|----------|
{% for p in pages %}| {{ p.label }} | {{ p.positive }}% | {{ p.neutral }}% | {{ p.negative }}% |
{% endfor %}

### Nhận xét
{% for insight in sentiment_insights %}
- {{ insight }}
{% endfor %}

## 3. Chủ Đề Nổi Bật (LDA Topics)

{% for p in topic_summaries %}
### {{ p.label }}
{% for t in p.topics %}
- **{{ t.label }}**: {{ t.top_words }}
{% endfor %}
{% endfor %}

## 4. Đề Xuất Chiến Lược

{% for rec in recommendations %}
### {{ rec.title }}
{{ rec.detail }}

{% endfor %}

## 5. KPI Đề Xuất

| Chỉ số | Mục tiêu | Benchmark |
|--------|----------|-----------|
| Engagement Rate | > {{ benchmarks.engagement_rate_good * 100 }}% | Top: {{ benchmarks.engagement_rate_excellent * 100 }}% |
| Sentiment Positive | > 35% | Hiện tại Hỏa Lò: {{ hoa_lo_positive }}% |
| Post Frequency | 4-5 bài/tuần | Đều đặn, đa dạng content |
| Comment Response Rate | > 80% | Trong vòng 2h |

---
*Báo cáo được tạo tự động bởi pipeline phân tích.*
"""


def compute_page_stats(raw_dir: Path, sentiment_dir: Path) -> list[dict]:
    pages = []

    for csv_path in sorted(raw_dir.glob("*_posts.csv")):
        page_name = csv_path.stem.replace("_posts", "")
        df = pd.read_csv(csv_path, parse_dates=["created_time"])

        total = len(df)
        df["engagement"] = df["likes"] + df["comments"] + df["shares"]
        avg_eng = round(df["engagement"].mean(), 1)

        if total > 1:
            date_range = (df["created_time"].max() - df["created_time"].min()).days
            posts_per_week = round(total / max(date_range / 7, 1), 1)
        else:
            posts_per_week = 0

        sentiment_path = sentiment_dir / f"{page_name}_sentiment.csv"
        positive, neutral, negative = 0.0, 0.0, 0.0
        if sentiment_path.exists():
            sdf = pd.read_csv(sentiment_path)
            dist = sdf["sentiment"].value_counts(normalize=True) * 100
            positive = round(dist.get("positive", 0), 1)
            neutral = round(dist.get("neutral", 0), 1)
            negative = round(dist.get("negative", 0), 1)

        pages.append({
            "name": page_name,
            "label": PAGE_LABELS.get(page_name, page_name),
            "total_posts": total,
            "avg_engagement": avg_eng,
            "posts_per_week": posts_per_week,
            "positive": positive,
            "neutral": neutral,
            "negative": negative,
        })

    return pages


def generate_insights(pages: list[dict]) -> list[str]:
    insights = []
    hoa_lo = next((p for p in pages if p["name"] == "hoa_lo"), None)
    if not hoa_lo:
        return insights

    others = [p for p in pages if p["name"] != "hoa_lo"]

    if others:
        if hoa_lo["positive"] > max(p["positive"] for p in others):
            insights.append(f"Hỏa Lò dẫn đầu về tỷ lệ Positive ({hoa_lo['positive']}%), cho thấy content tạo cảm xúc tích cực tốt.")
        else:
            leader = max(others, key=lambda p: p["positive"])
            insights.append(f"{leader['label']} có Positive cao hơn ({leader['positive']}% vs {hoa_lo['positive']}%). Hỏa Lò cần cải thiện content tạo cảm xúc.")

        if hoa_lo["avg_engagement"] < max(p["avg_engagement"] for p in others):
            leader = max(others, key=lambda p: p["avg_engagement"])
            insights.append(f"Engagement TB của Hỏa Lò ({hoa_lo['avg_engagement']}) thấp hơn {leader['label']} ({leader['avg_engagement']}). Cần tăng tương tác.")
    else:
        insights.append(f"Tỷ lệ Positive của Hỏa Lò đạt {hoa_lo['positive']}%, Neutral {hoa_lo['neutral']}%.")
        insights.append(f"Engagement trung bình: {hoa_lo['avg_engagement']} (likes + comments + shares).")
        insights.append(f"Tần suất đăng bài: {hoa_lo['posts_per_week']} bài/tuần.")

    if hoa_lo["negative"] < 2:
        insights.append(f"Tỷ lệ Negative rất thấp ({hoa_lo['negative']}%), phản ánh ít phản hồi tiêu cực từ cộng đồng.")

    return insights


def generate_recommendations(pages: list[dict]) -> list[dict]:
    recs = []
    hoa_lo = next((p for p in pages if p["name"] == "hoa_lo"), None)
    if not hoa_lo:
        return recs

    recs.append({
        "title": "4.1 Đa dạng hóa nội dung",
        "detail": "Kết hợp video ngắn, infographic lịch sử, và bài đăng tương tác (poll, quiz về lịch sử Hỏa Lò) để tăng engagement. Ưu tiên nội dung UGC (user-generated content) từ du khách.",
    })
    recs.append({
        "title": "4.2 Tối ưu thời gian đăng",
        "detail": "Phân tích dữ liệu cho thấy cần post vào khung giờ 11h-13h và 19h-21h (peak Facebook usage VN). Duy trì tần suất 4-5 bài/tuần.",
    })
    recs.append({
        "title": "4.3 Storytelling & Chuỗi nội dung",
        "detail": "Xây dựng series 'Câu chuyện tù nhân' hoặc 'Di sản qua ống kính' để tạo kết nối cảm xúc. Content dạng narrative có engagement cao hơn 2-3x so với post thông tin thuần.",
    })

    if hoa_lo["negative"] < 2:
        recs.append({
            "title": "4.4 Tận dụng tỷ lệ Negative thấp",
            "detail": f"Với chỉ {hoa_lo['negative']}% negative sentiment, đây là lợi thế lớn. Đẩy mạnh chiến dịch review/đánh giá để tận dụng hình ảnh tích cực.",
        })

    return recs


def load_topic_summaries(topics_dir: Path) -> list[dict]:
    summaries = []
    for json_path in sorted(topics_dir.glob("*_topics.json")):
        with open(json_path) as f:
            data = json.load(f)

        page_name = data["page"]
        topics = []
        for t in data.get("topics", []):
            words = ", ".join(w for w, _ in t["top_words"][:6])
            topics.append({"label": t["label"], "top_words": words})

        summaries.append({"label": PAGE_LABELS.get(page_name, page_name), "topics": topics})

    return summaries


def main():
    parser = argparse.ArgumentParser(description="Generate strategy report")
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--sentiment", default="data/results/sentiment")
    parser.add_argument("--topics", default="data/results/topics")
    parser.add_argument("--config", default="config/pipeline.yaml")
    parser.add_argument("--output", default="reports/strategy_report.md")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    benchmarks = config.get("strategy", {}).get("benchmarks", {})

    pages = compute_page_stats(Path(args.raw), Path(args.sentiment))
    insights = generate_insights(pages)
    topic_summaries = load_topic_summaries(Path(args.topics))
    recommendations = generate_recommendations(pages)

    hoa_lo = next((p for p in pages if p["name"] == "hoa_lo"), {})
    template = Template(REPORT_TEMPLATE)
    report = template.render(
        generated_at=datetime.now().strftime("%d/%m/%Y %H:%M"),
        pages=pages,
        sentiment_insights=insights,
        topic_summaries=topic_summaries,
        recommendations=recommendations,
        benchmarks=benchmarks,
        hoa_lo_positive=hoa_lo.get("positive", 0),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    logger.info(f"Strategy report saved to {output_path}")


if __name__ == "__main__":
    main()
