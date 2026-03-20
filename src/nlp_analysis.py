"""
NLP Analysis Pipeline — reproduces HoaLo_NLP_Analysis.ipynb
Input: posts_cleaned.csv (from clean_data.py)
Output: enriched CSV, JSON report, 4 PNG charts
Steps: preprocess → LDA topics (gensim) → sentiment (rule-based) → keywords → cross-analysis
"""
import os, re, json, logging, argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from underthesea import word_tokenize
from gensim import corpora
from gensim.models import LdaModel
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STOPWORDS = set([
    "và", "của", "có", "được", "trong", "là", "cho", "với", "từ", "một",
    "các", "này", "những", "đã", "để", "cũng", "người", "không", "nhưng",
    "về", "đến", "hay", "lại", "thì", "vào", "ra", "năm", "ngày", "tại",
    "như", "theo", "họ", "nó", "bởi", "khi", "nếu", "đang", "sẽ",
    "tôi", "bạn", "anh", "chị", "em", "chúng_tôi", "chúng_ta", "mình",
    "ta", "ông", "bà", "cô", "chú", "cậu",
    "trên", "dưới", "giữa", "ngoài", "sau", "trước", "bên", "cùng",
    "qua", "đối_với", "nhờ", "tới", "khỏi",
    "mà", "rằng", "vì", "nên", "hoặc", "hay_là", "song",
    "do", "bởi_vì", "vì_vậy", "tuy", "dù",
    "gì", "ai", "đâu", "nào", "sao", "thế_nào", "bao_giờ",
    "mỗi", "mọi", "cả", "toàn", "tất_cả", "nhiều", "ít", "vài",
    "đủ", "thêm", "nữa", "khác",
    "làm", "đi", "nói", "thấy", "biết", "muốn", "cần", "phải",
    "bị", "đưa", "lấy", "có_thể",
    "rất", "quá", "khá", "hơn", "nhất", "chỉ", "đều",
    "còn", "vẫn", "vừa", "mới", "luôn", "thường",
    "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín", "mười",
    "cái", "chiếc", "con", "quả", "trái", "bức", "tờ", "cuốn",
    "chưa", "rồi",
])

POSITIVE_WORDS = set([
    "tuyệt", "đẹp", "hay", "tốt", "thích", "yêu", "xuất_sắc", "ấn_tượng",
    "tuyệt_vời", "hoàn_hảo", "đáng", "great", "good", "excellent",
    "amazing", "wonderful", "love", "nice",
    "tự_hào", "vinh_quang", "anh_dũng", "kiên_cường", "vẻ_vang",
    "cảm_động", "xúc_động", "đáng_nhớ", "phi_thường", "vĩ_đại",
])

NEGATIVE_WORDS = set([
    "tệ", "xấu", "kém", "không_tốt", "thất_vọng", "buồn", "dở",
    "không_nên", "bad", "poor", "terrible", "disappointed", "waste",
    "thiếu", "nhàm", "tẻ_nhạt", "chán",
    "đau_thương", "khổ_cực", "tàn_bạo", "đau_khổ", "bi_thảm",
])

HOALO_KEYWORDS = ["tù", "giam", "tù_binh", "nhà_tù", "lịch_sử", "cách_mạng",
                   "chiến_sĩ", "hiện_vật", "triển_lãm", "tham_quan", "di_tích", "bảo_tàng"]

SENT_COLORS = {"positive": "#4CAF50", "neutral": "#FFC107", "negative": "#f44336"}


def preprocess(df_text):
    """Clean text + tokenize Vietnamese"""
    def clean(text):
        if pd.isna(text) or text == "":
            return ""
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"\S+@\S+", "", text)
        text = re.sub(r"[^a-zA-ZÀ-ỹ\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def tokenize(text):
        if not text:
            return []
        try:
            tokens = word_tokenize(text, format="text").split()
            return [t for t in tokens if t not in STOPWORDS and len(t) > 1]
        except Exception:
            return text.split()

    df_text["text_clean"] = df_text["text"].apply(clean)
    df_text["tokens"] = df_text["text_clean"].apply(tokenize)
    df_text["token_count"] = df_text["tokens"].apply(len)
    logger.info(f"Preprocessed: avg {df_text['token_count'].mean():.1f} tokens/post, {len(set(t for toks in df_text['tokens'] for t in toks)):,} unique")
    return df_text


def run_lda(df_text, num_topics=7):
    """LDA topic modeling using gensim (matches notebook)"""
    min_tokens = 3
    valid_mask = df_text["token_count"] >= min_tokens
    df_text["original_index"] = range(len(df_text))
    texts = df_text[valid_mask]["tokens"].tolist()
    valid_indices = df_text[valid_mask]["original_index"].tolist()

    logger.info(f"LDA: {valid_mask.sum()} valid docs (>={min_tokens} tokens), {(~valid_mask).sum()} too short → Topic -1")

    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=2, no_above=0.8)
    corpus = [dictionary.doc2bow(t) for t in texts]

    model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics,
                     random_state=42, passes=10, alpha="auto", per_word_topics=True)

    topic_labels = {}
    for idx, topic in model.print_topics(-1, num_words=10):
        words = [w.split("*")[1].replace('"', "").strip() for w in topic.split("+")]
        topic_labels[idx] = ", ".join(words[:5])
        logger.info(f"  Topic {idx}: {topic_labels[idx]}")

    df_text["topic_id"] = -1
    df_text["topic_prob"] = 0.0
    for idx, bow in zip(valid_indices, corpus):
        probs = model.get_document_topics(bow)
        if probs:
            best = max(probs, key=lambda x: x[1])
            df_text.loc[df_text["original_index"] == idx, "topic_id"] = best[0]
            df_text.loc[df_text["original_index"] == idx, "topic_prob"] = best[1]

    df_text["topic_label"] = df_text["topic_id"].apply(lambda x: topic_labels.get(x, "Too Short") if x >= 0 else "Too Short")
    logger.info(f"Topic distribution:\n{df_text['topic_id'].value_counts().sort_index()}")
    return df_text, topic_labels, model, dictionary, corpus


def run_sentiment(df_text):
    """Rule-based sentiment (matches notebook — NOT underthesea classifier)"""
    def classify(tokens):
        if not tokens:
            return "neutral", 0
        pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
        neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
        score = pos - neg
        if pos > neg:
            return "positive", score
        elif neg > pos:
            return "negative", score
        return "neutral", score

    results = df_text["tokens"].apply(classify)
    df_text["sentiment"] = results.apply(lambda x: x[0])
    df_text["sentiment_score"] = results.apply(lambda x: x[1])

    dist = df_text["sentiment"].value_counts()
    for s, c in dist.items():
        logger.info(f"  {s}: {c} ({c/len(df_text)*100:.1f}%)")
    return df_text


def run_keywords(df_text):
    """TF-IDF + word frequency + Hỏa Lò specific keywords"""
    corpus_text = df_text["tokens"].apply(lambda x: " ".join(x)).tolist()
    tfidf = TfidfVectorizer(max_features=100, ngram_range=(1, 2), min_df=3, max_df=0.7)
    matrix = tfidf.fit_transform(corpus_text)
    features = tfidf.get_feature_names_out()
    scores = matrix.sum(axis=0).A1
    top_idx = scores.argsort()[-30:][::-1]
    top_tfidf = [(features[i], float(scores[i])) for i in top_idx]

    all_tokens = [t for toks in df_text["tokens"] for t in toks]
    word_freq = Counter(all_tokens)
    top_words = word_freq.most_common(50)

    hoalo_kw = {}
    for kw in HOALO_KEYWORDS:
        count = sum(1 for toks in df_text["tokens"] if kw in toks)
        if count > 0:
            hoalo_kw[kw] = count

    return top_tfidf, top_words, hoalo_kw


def plot_topics(df_text, topic_labels, out_dir):
    df_valid = df_text[df_text["topic_id"] >= 0]
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    counts = df_valid["topic_id"].value_counts().sort_index()
    colors = plt.cm.Set3(range(len(counts)))

    axes[0, 0].bar(range(len(counts)), counts.values, color=colors)
    for i, v in enumerate(counts.values):
        axes[0, 0].text(i, v, str(v), ha="center", va="bottom")
    axes[0, 0].set_title("Topic Distribution", fontweight="bold")
    axes[0, 0].set_xlabel("Topic ID")
    axes[0, 0].grid(axis="y", alpha=0.3)

    axes[0, 1].pie(counts.values, labels=[f"T{i}\n({counts[i]})" for i in counts.index], autopct="%1.1f%%", colors=colors)
    axes[0, 1].set_title("Topic %", fontweight="bold")

    eng = df_valid.groupby("topic_id")["engagement_total"].mean().sort_index()
    axes[1, 0].bar(range(len(eng)), eng.values, color=colors)
    for i, v in enumerate(eng.values):
        axes[1, 0].text(i, v, f"{v:.0f}", ha="center", va="bottom")
    axes[1, 0].set_title("Avg Engagement by Topic", fontweight="bold")
    axes[1, 0].grid(axis="y", alpha=0.3)

    df_time = df_valid[df_valid["year_month"].notna()]
    if not df_time.empty:
        tt = df_time.groupby(["year_month", "topic_id"]).size().unstack(fill_value=0)
        tt.plot(kind="area", stacked=True, ax=axes[1, 1], alpha=0.7)
        axes[1, 1].set_title("Topics Over Time", fontweight="bold")
        plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.suptitle("TOPIC MODELING ANALYSIS", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "nlp_01_topic_modeling.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_sentiment(df_text, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    sc = df_text["sentiment"].value_counts()
    colors = [SENT_COLORS.get(s, "gray") for s in sc.index]
    axes[0, 0].pie(sc.values, labels=sc.index, autopct="%1.1f%%", colors=colors)
    axes[0, 0].set_title("Sentiment Distribution", fontweight="bold")

    df_t = df_text[df_text["year_month"].notna()]
    st = df_t.groupby(["year_month", "sentiment"]).size().unstack(fill_value=0)
    st_pct = st.div(st.sum(axis=1), axis=0) * 100
    st_pct.plot(kind="bar", stacked=True, ax=axes[0, 1], color=[SENT_COLORS.get(c, "gray") for c in st_pct.columns])
    axes[0, 1].set_title("Sentiment Over Time", fontweight="bold")
    plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45, ha="right")

    se = df_text.groupby("sentiment")["engagement_total"].mean()
    axes[1, 0].bar(range(len(se)), se.values, color=[SENT_COLORS.get(s, "gray") for s in se.index])
    axes[1, 0].set_xticks(range(len(se)))
    axes[1, 0].set_xticklabels(se.index)
    axes[1, 0].set_title("Engagement by Sentiment", fontweight="bold")

    axes[1, 1].hist(df_text["sentiment_score"], bins=20, color="steelblue", edgecolor="black", alpha=0.7)
    axes[1, 1].axvline(0, color="red", linestyle="--", linewidth=2)
    axes[1, 1].set_title("Sentiment Score Distribution", fontweight="bold")

    plt.suptitle("SENTIMENT ANALYSIS", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "nlp_02_sentiment_analysis.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_keywords(top_tfidf, top_words, hoalo_kw, df_text, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    all_text = " ".join(df_text["tokens"].apply(lambda x: " ".join(x)))
    wc = WordCloud(width=800, height=400, background_color="white", colormap="viridis", max_words=100).generate(all_text)
    axes[0, 0].imshow(wc, interpolation="bilinear")
    axes[0, 0].axis("off")
    axes[0, 0].set_title("Word Cloud", fontweight="bold")

    w20 = top_words[:20]
    axes[0, 1].barh(range(len(w20)), [c for _, c in w20], color="steelblue", alpha=0.7)
    axes[0, 1].set_yticks(range(len(w20)))
    axes[0, 1].set_yticklabels([w for w, _ in w20])
    axes[0, 1].set_title("Top 20 Words", fontweight="bold")
    axes[0, 1].invert_yaxis()

    t15 = top_tfidf[:15]
    axes[1, 0].barh(range(len(t15)), [s for _, s in t15], color="coral", alpha=0.7)
    axes[1, 0].set_yticks(range(len(t15)))
    axes[1, 0].set_yticklabels([w for w, _ in t15])
    axes[1, 0].set_title("Top 15 TF-IDF", fontweight="bold")
    axes[1, 0].invert_yaxis()

    if hoalo_kw:
        items = sorted(hoalo_kw.items(), key=lambda x: x[1], reverse=True)
        axes[1, 1].bar(range(len(items)), [v for _, v in items], color="purple", alpha=0.7)
        axes[1, 1].set_xticks(range(len(items)))
        axes[1, 1].set_xticklabels([k for k, _ in items], rotation=45, ha="right")
        axes[1, 1].set_title("Hỏa Lò Keywords", fontweight="bold")

    plt.suptitle("KEYWORD ANALYSIS", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "nlp_03_keyword_analysis.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_cross_analysis(df_text, out_dir):
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])
    ct = pd.crosstab(df_text["topic_id"], df_text["sentiment"])
    sns.heatmap(ct, annot=True, fmt="d", cmap="YlGnBu", ax=ax1)
    ax1.set_title("Topic × Sentiment", fontweight="bold", fontsize=14)

    df_v = df_text[df_text["topic_id"] >= 0]
    te = df_v.groupby("topic_id")["engagement_total"].mean().sort_values(ascending=False)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.barh(range(len(te)), te.values, color="green", alpha=0.7)
    ax2.set_yticks(range(len(te)))
    ax2.set_yticklabels([f"Topic {int(i)}" for i in te.index])
    ax2.set_title("Engagement by Topic", fontweight="bold")

    tc = df_text["topic_id"].value_counts().sort_index()
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.bar(range(len(tc)), tc.values, color="steelblue", alpha=0.7)
    ax3.set_xticks(range(len(tc)))
    ax3.set_xticklabels([f"T{int(i)}" for i in tc.index])
    ax3.set_title("Posts by Topic", fontweight="bold")

    ax4 = fig.add_subplot(gs[1, 2])
    cp = pd.crosstab(df_text["topic_id"], df_text["sentiment"], normalize="index") * 100
    cp.plot(kind="bar", stacked=True, ax=ax4, color=[SENT_COLORS.get(c, "gray") for c in cp.columns])
    ax4.set_title("Sentiment % by Topic", fontweight="bold")
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=0)

    ter = df_v.groupby("topic_id")["engagement_total"].mean().sort_values(ascending=True)
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.barh(range(len(ter)), ter.values, color=plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(ter))))
    ax5.set_yticks(range(len(ter)))
    ax5.set_yticklabels([f"Topic {int(i)}" for i in ter.index])
    ax5.set_title("Topic Ranking", fontweight="bold")

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.scatter(df_text["text_length"], df_text["engagement_total"], alpha=0.3, s=20)
    corr = df_text[["text_length", "engagement_total"]].corr().iloc[0, 1]
    ax6.text(0.05, 0.95, f"Corr: {corr:.3f}", transform=ax6.transAxes, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax6.set_title("Engagement vs Length", fontweight="bold")

    se = df_text.groupby("sentiment")["engagement_total"].mean()
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.bar(range(len(se)), se.values, color=[SENT_COLORS.get(s, "gray") for s in se.index])
    ax7.set_xticks(range(len(se)))
    ax7.set_xticklabels(se.index)
    ax7.set_title("Engagement by Sentiment", fontweight="bold")

    plt.suptitle("COMPREHENSIVE NLP INSIGHTS — Hỏa Lò", fontsize=18, fontweight="bold")
    plt.savefig(out_dir / "nlp_04_comprehensive_insights.png", dpi=200, bbox_inches="tight")
    plt.close()


def export_results(df_text, topic_labels, top_tfidf, top_words, hoalo_kw, out_dir):
    cols = ["postId", "date", "text", "text_length", "topic_id", "topic_label", "topic_prob",
            "sentiment", "sentiment_score", "engagement_total", "reactions_total", "comment_count", "share_count"]
    available = [c for c in cols if c in df_text.columns]
    df_text[available].to_csv(out_dir / "posts_nlp_enriched.csv", index=False, encoding="utf-8-sig")

    dist = df_text["sentiment"].value_counts()
    df_valid = df_text[df_text["topic_id"] >= 0]
    topic_stats = df_valid.groupby("topic_id").agg(
        engagement_mean=("engagement_total", "mean"), post_count=("topic_id", "count")).round(1)
    best_tid = topic_stats["engagement_mean"].idxmax() if not topic_stats.empty else -1
    se = df_text.groupby("sentiment")["engagement_total"].mean()
    corr = float(df_text[["text_length", "engagement_total"]].corr().iloc[0, 1])
    sc = float(df_text[["sentiment_score", "engagement_total"]].corr().iloc[0, 1]) if "sentiment_score" in df_text.columns else 0

    report = {
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {"total_posts": len(df_text), "avg_text_length": round(float(df_text["text_length"].mean()), 1),
                     "total_tokens": int(df_text["token_count"].sum())},
        "topics": {"num_topics": len(topic_labels), "labels": {str(k): v for k, v in topic_labels.items()},
                    "best_topic": int(best_tid),
                    "best_topic_engagement": float(topic_stats.loc[best_tid, "engagement_mean"]) if best_tid >= 0 else 0},
        "sentiment": {s: {"count": int(dist.get(s, 0)), "pct": round(float(dist.get(s, 0) / len(df_text) * 100), 1)}
                       for s in ["positive", "neutral", "negative"]},
        "keywords": {"top_20_words": [(w, int(c)) for w, c in top_words[:20]],
                      "top_15_tfidf": [(k, round(s, 3)) for k, s in top_tfidf[:15]],
                      "hoalo_keywords": {str(k): int(v) for k, v in hoalo_kw.items()}},
        "insights": {"engagement_sentiment_corr": round(sc, 3), "engagement_length_corr": round(corr, 3),
                      "best_sentiment": str(se.idxmax()), "best_sentiment_engagement": round(float(se.max()), 1)},
    }
    with open(out_dir / "nlp_analysis_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Exported enriched CSV ({len(df_text)} rows) + report JSON")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/cleaned")
    parser.add_argument("--output", default="data/results")
    parser.add_argument("--figures", default="reports/figures")
    parser.add_argument("--topics", type=int, default=7)
    parser.add_argument("--config", default="config/pipeline.yaml")
    args = parser.parse_args()

    inp, out, figs = Path(args.input), Path(args.output), Path(args.figures)
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    csv_path = inp / "posts_cleaned.csv"
    if not csv_path.exists():
        logger.error(f"posts_cleaned.csv not found in {inp}")
        return

    df = pd.read_csv(csv_path, encoding="utf-8")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df_text = df[df["has_text"] == True].copy()
    logger.info(f"Loaded {len(df)} posts, {len(df_text)} with text")

    logger.info("=== Step 1: Preprocessing ===")
    df_text = preprocess(df_text)

    logger.info("=== Step 2: LDA Topic Modeling ===")
    df_text, topic_labels, model, dictionary, corpus = run_lda(df_text, num_topics=args.topics)

    logger.info("=== Step 3: Sentiment Analysis ===")
    df_text = run_sentiment(df_text)

    logger.info("=== Step 4: Keyword Extraction ===")
    top_tfidf, top_words, hoalo_kw = run_keywords(df_text)

    logger.info("=== Step 5: Visualizations ===")
    plot_topics(df_text, topic_labels, figs)
    plot_sentiment(df_text, figs)
    plot_keywords(top_tfidf, top_words, hoalo_kw, df_text, figs)
    plot_cross_analysis(df_text, figs)

    logger.info("=== Step 6: Export ===")
    report = export_results(df_text, topic_labels, top_tfidf, top_words, hoalo_kw, out)

    logger.info("NLP Analysis complete!")
    logger.info(f"  Posts: {report['dataset']['total_posts']}")
    logger.info(f"  Topics: {report['topics']['num_topics']}, best: Topic {report['topics']['best_topic']}")
    for s in ["positive", "neutral", "negative"]:
        logger.info(f"  {s}: {report['sentiment'][s]['pct']}%")


if __name__ == "__main__":
    main()
