"""Tests for the pipeline modules."""
import pytest
import pandas as pd

import sys
sys.path.insert(0, ".")

from src.preprocessing import clean_text, load_stopwords


class TestCleanText:
    def test_removes_urls(self):
        config = {"remove_urls": True, "remove_emojis": True, "lowercase": True}
        result = clean_text("Check this https://example.com link", config)
        assert "https" not in result
        assert "example" not in result

    def test_removes_emojis(self):
        config = {"remove_urls": True, "remove_emojis": True, "lowercase": True}
        result = clean_text("Great post 🎉🔥", config)
        assert "🎉" not in result

    def test_lowercase(self):
        config = {"remove_urls": True, "remove_emojis": True, "lowercase": True}
        result = clean_text("HELLO World", config)
        assert result == "hello world"

    def test_empty_input(self):
        config = {"remove_urls": True, "remove_emojis": True, "lowercase": True}
        assert clean_text("", config) == ""
        assert clean_text(None, config) == ""

    def test_preserves_vietnamese(self):
        config = {"remove_urls": True, "remove_emojis": True, "lowercase": True}
        result = clean_text("Nhà tù Hỏa Lò rất đẹp", config)
        assert "nhà" in result
        assert "hỏa" in result


class TestLoadStopwords:
    def test_load_from_file(self, tmp_path):
        sw_file = tmp_path / "stopwords.txt"
        sw_file.write_text("và\ncủa\nlà\n# comment\n\nnhưng\n", encoding="utf-8")
        result = load_stopwords(str(sw_file))
        assert "và" in result
        assert "của" in result
        assert "# comment" not in result
        assert "" not in result

    def test_missing_file(self):
        result = load_stopwords("/nonexistent/path.txt")
        assert result == set()


class TestSentiment:
    def test_classify_empty(self):
        from src.sentiment import classify_sentiment
        assert classify_sentiment("") == "neutral"
        assert classify_sentiment(None) == "neutral"

    def test_post_processing_positive(self):
        from src.sentiment import apply_post_processing
        row = pd.Series({
            "text_tokenized": "tuyệt vời đẹp hay thích",
            "sentiment_raw": "neutral"
        })
        result = apply_post_processing(row)
        assert result == "positive"

    def test_post_processing_keeps_neutral(self):
        from src.sentiment import apply_post_processing
        row = pd.Series({
            "text_tokenized": "bài đăng thông tin chung",
            "sentiment_raw": "neutral"
        })
        result = apply_post_processing(row)
        assert result == "neutral"


class TestTopicModeling:
    def test_tfidf_basic(self):
        from src.topic_modeling import run_tfidf
        config = {"nlp": {"tfidf": {"max_features": 100, "ngram_range": [1, 1], "min_df": 1, "max_df": 1.0}}}
        texts = ["hello world", "world peace", "hello peace world"]
        matrix, features, _ = run_tfidf(texts, config)
        assert matrix.shape[0] == 3
        assert len(features) > 0


class TestScraper:
    def test_normalize_items(self):
        from src.scraper import normalize_items
        items = [
            {"postId": "123", "text": "Test post", "time": "2025-01-01T10:00:00", "type": "photo", "likes": 10, "comments": 2, "shares": 1},
            {"postId": "456", "text": "", "time": "2025-01-02T10:00:00", "type": "status", "likes": {"total": 5}, "comments": {"count": 1}, "shares": 0},
        ]
        df = normalize_items(items)
        assert len(df) == 2
        assert df.iloc[0]["post_id"] == "456"  # sorted desc by time
        assert df.iloc[1]["likes"] == 10

    def test_normalize_empty(self):
        from src.scraper import normalize_items
        df = normalize_items([])
        assert len(df) == 0
