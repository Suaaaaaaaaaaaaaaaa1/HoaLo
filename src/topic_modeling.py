"""
LDA topic modeling + TF-IDF analysis for Vietnamese Facebook posts.
"""
import os
import json
import logging
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_tfidf(texts: list[str], config: dict) -> tuple:
    tfidf_cfg = config.get("nlp", {}).get("tfidf", {})

    vectorizer = TfidfVectorizer(
        max_features=tfidf_cfg.get("max_features", 5000),
        ngram_range=tuple(tfidf_cfg.get("ngram_range", [1, 2])),
        min_df=tfidf_cfg.get("min_df", 3),
        max_df=tfidf_cfg.get("max_df", 0.85),
    )

    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    logger.info(f"TF-IDF matrix: {tfidf_matrix.shape[0]} docs × {tfidf_matrix.shape[1]} features")
    return tfidf_matrix, feature_names, vectorizer


def run_lda(tfidf_matrix, feature_names, config: dict) -> tuple:
    lda_cfg = config.get("nlp", {}).get("lda", {})

    n_topics = lda_cfg.get("n_topics", 5)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=lda_cfg.get("max_iter", 30),
        random_state=lda_cfg.get("random_state", 42),
        learning_method=lda_cfg.get("learning_method", "online"),
    )

    doc_topic_dist = lda.fit_transform(tfidf_matrix)

    topics = []
    for idx, component in enumerate(lda.components_):
        top_indices = component.argsort()[-15:][::-1]
        top_words = [(feature_names[i], round(float(component[i]), 4)) for i in top_indices]
        topics.append({
            "topic_id": idx,
            "top_words": top_words,
            "label": f"Topic_{idx}",
        })
        word_str = ", ".join(f"{w}({s})" for w, s in top_words[:8])
        logger.info(f"  Topic {idx}: {word_str}")

    return lda, doc_topic_dist, topics


def get_top_tfidf_terms(tfidf_matrix, feature_names, top_n: int = 30) -> list[dict]:
    mean_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
    top_indices = mean_scores.argsort()[-top_n:][::-1]
    return [
        {"term": feature_names[i], "mean_tfidf": round(float(mean_scores[i]), 6)}
        for i in top_indices
    ]


def main():
    parser = argparse.ArgumentParser(description="Run TF-IDF and LDA topic modeling")
    parser.add_argument("--input", default="data/processed")
    parser.add_argument("--output", default="data/results/topics")
    parser.add_argument("--config", default="config/pipeline.yaml")
    parser.add_argument("--n-topics", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.n_topics:
        config.setdefault("nlp", {}).setdefault("lda", {})["n_topics"] = args.n_topics

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(input_dir.glob("*_processed.csv"))
    if not csv_files:
        logger.error(f"No processed CSV files found in {input_dir}")
        return

    for csv_path in csv_files:
        page_name = csv_path.name.replace("_processed.csv", "")
        logger.info(f"=== Topic Modeling: {page_name} ===")

        df = pd.read_csv(csv_path)
        texts = df["text_tokenized"].dropna().tolist()
        if len(texts) < 10:
            logger.warning(f"Only {len(texts)} texts for {page_name}, skipping")
            continue

        tfidf_matrix, feature_names, vectorizer = run_tfidf(texts, config)

        logger.info("Running LDA...")
        lda_model, doc_topic_dist, topics = run_lda(tfidf_matrix, feature_names, config)

        top_terms = get_top_tfidf_terms(tfidf_matrix, feature_names)

        topics_path = output_dir / f"{page_name}_topics.json"
        with open(topics_path, "w", encoding="utf-8") as f:
            json.dump({"page": page_name, "topics": topics, "top_tfidf_terms": top_terms}, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved topics to {topics_path}")

        df_topics = df.copy()
        for i in range(doc_topic_dist.shape[1]):
            df_topics[f"topic_{i}_weight"] = 0.0
        valid_idx = df["text_tokenized"].dropna().index
        for col_i in range(doc_topic_dist.shape[1]):
            df_topics.loc[valid_idx, f"topic_{col_i}_weight"] = doc_topic_dist[:, col_i]
        df_topics["dominant_topic"] = df_topics[[f"topic_{i}_weight" for i in range(doc_topic_dist.shape[1])]].idxmax(axis=1)

        doc_topics_path = output_dir / f"{page_name}_doc_topics.csv"
        df_topics.to_csv(doc_topics_path, index=False, encoding="utf-8-sig")
        logger.info(f"Saved document-topic matrix to {doc_topics_path}")


if __name__ == "__main__":
    main()
