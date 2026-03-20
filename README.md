# 🏛️ Hỏa Lò Facebook Analysis

Phân tích chiến lược Social Media Marketing của **Di tích Nhà tù Hỏa Lò** trên Facebook.

> **Dự án học thuật** — Khoa Hệ thống Thông tin, Trường Đại học Kinh tế - Luật (UEL), ĐHQG-HCM.

---

## Pipeline Overview

```
┌────────────┐    ┌──────────────┐    ┌───────────┐    ┌────────┐    ┌──────────┐    ┌───────────┐    ┌───────┐
│   Apify    │───▶│ Preprocessor │───▶│ Sentiment │───▶│  LDA   │───▶│ Visualize│───▶│ Strategy  │───▶│ Email │
│  Scraper   │    │ (underthesea)│    │ Analysis  │    │ Topics │    │ (charts) │    │ (report)  │    │Sender │
└────────────┘    └──────────────┘    └───────────┘    └────────┘    └──────────┘    └───────────┘    └───────┘
```

Thu thập dữ liệu bằng **Apify Facebook Posts Scraper** (`apify/facebook-posts-scraper`), xử lý NLP tiếng Việt, và tự động tạo báo cáo chiến lược.

## Kết quả Sentiment mới nhất (v2.1)

| Sentiment | v1.0 | v2.1 | Thay đổi |
|-----------|------|------|----------|
| Neutral   | 64.7% | **67.0%** | +2.3 |
| Positive  | 29.1% | **32.4%** | +3.3 |
| Negative  | 6.2%  | **0.6%**  | −5.6 |

---

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/your-username/hoa-lo-facebook-analysis.git
cd hoa-lo-facebook-analysis

python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### 2. Cấu hình

```bash
cp .env.example .env
```

Sửa `.env`:
- `APIFY_API_TOKEN` — Lấy từ [Apify Console → Settings → Integrations](https://console.apify.com/account/integrations)
- `EMAIL_SENDER` / `EMAIL_PASSWORD` / `EMAIL_RECIPIENTS` — Nếu muốn gửi báo cáo email

### 3. Chạy toàn bộ pipeline

```bash
# Chạy tất cả 7 bước
python src/pipeline.py

# Bỏ qua scrape (đã có data)
python src/pipeline.py --start-from preprocess

# Chỉ chạy 1 bước
python src/pipeline.py --only sentiment

# Bỏ qua email
python src/pipeline.py --skip email

# Dry run
python src/pipeline.py --dry-run
```

### 4. Chạy từng bước riêng lẻ

```bash
# Scrape via Apify
python src/scraper.py --output data/raw

# Preprocess
python src/preprocessing.py --input data/raw --output data/processed

# Sentiment
python src/sentiment.py --input data/processed --output data/results/sentiment

# Topic Modeling
python src/topic_modeling.py --input data/processed --output data/results/topics

# Visualizations
python src/visualize.py --sentiment data/results/sentiment --topics data/results/topics --output reports/figures

# Strategy Report
python src/strategy.py --raw data/raw --sentiment data/results/sentiment --topics data/results/topics --output reports/strategy_report.md

# Email (dry run)
python src/email_sender.py --dry-run
```

---

## Cấu trúc thư mục

```
hoa-lo-facebook-analysis/
├── .github/workflows/ci.yml    # GitHub Actions pipeline (7 jobs)
├── config/
│   └── pipeline.yaml            # Toàn bộ config
├── data/
│   ├── raw/                     # CSV + JSON gốc từ Apify
│   ├── processed/               # Sau preprocessing
│   ├── results/
│   │   ├── sentiment/           # Kết quả sentiment analysis
│   │   └── topics/              # Kết quả LDA + TF-IDF
│   └── vietnamese_stopwords.txt
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_nlp.ipynb
├── src/
│   ├── pipeline.py              # Main orchestrator
│   ├── scraper.py               # Apify Facebook Posts Scraper
│   ├── preprocessing.py         # Vietnamese text preprocessing
│   ├── sentiment.py             # Sentiment (underthesea + rules)
│   ├── topic_modeling.py        # TF-IDF + LDA
│   ├── visualize.py             # Charts (matplotlib/seaborn)
│   ├── strategy.py              # Strategy report (Jinja2)
│   └── email_sender.py          # SMTP email
├── tests/
│   └── test_pipeline.py
├── reports/
│   ├── figures/                 # Generated PNG charts
│   └── strategy_report.md
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## GitHub Actions

### Workflow (7 Jobs)

| # | Job | Trigger | Mô tả |
|---|-----|---------|--------|
| 1 | `lint` | push/PR | Ruff + notebook strip check |
| 2 | `test` | after lint | pytest + coverage |
| 3 | `scrape` | schedule/manual | **Apify** scrape page Hỏa Lò |
| 4 | `data-validation` | after test | Validate schema CSV |
| 5 | `nlp-pipeline` | after test+validation | Preprocessing → Sentiment → LDA |
| 6 | `build-report` | main only | Charts + strategy report |
| 7 | `send-email` | schedule/manual | Gửi report qua email |

### Secrets cần thiết

Thêm trong **Settings → Secrets and variables → Actions**:

| Secret | Mô tả |
|--------|--------|
| `APIFY_API_TOKEN` | Apify API token |
| `EMAIL_SENDER` | Gmail address |
| `EMAIL_PASSWORD` | Gmail App Password |
| `EMAIL_RECIPIENTS` | Comma-separated email list |

### Manual Trigger

Vào **Actions → Run workflow**:
- `skip_scrape`: `true` nếu đã có data (default)
- `send_email`: `true` để gửi report

### Scheduled

Tự động **thứ Hai hàng tuần 2:00 UTC** — scrape mới + full pipeline + email.

---

## NLP Stack

| Component | Technology |
|-----------|------------|
| Tokenization | underthesea |
| Stopwords | Custom Vietnamese (80+ words) |
| Vectorization | TF-IDF (sklearn), ngram=(1,2) |
| Topic Modeling | LDA (sklearn), 5 topics |
| Sentiment | underthesea + rule-based post-processing |

## Apify Setup

1. Tạo tài khoản tại [apify.com](https://apify.com)
2. Lấy API token từ **Console → Settings → Integrations**
3. Actor sử dụng: [`apify/facebook-posts-scraper`](https://apify.com/apify/facebook-posts-scraper)
4. Free plan cho ~500 posts, Starter plan ($29/tháng) cho nhiều hơn

---

## Tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=src --cov-report=term-missing
```

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 03/2026 | v2.1 | Chuyển sang Apify scraper. Sentiment recalculated (67/32.4/0.6). GitHub Actions CI/CD. |
| 02/2026 | v2.0 | NLP pipeline: LDA, TF-IDF, sentiment |
| 01/2026 | v1.0 | EDA. Sentiment v1: 64.7/29.1/6.2 |

---

**UEL — Khoa Hệ thống Thông tin — K22416C**
