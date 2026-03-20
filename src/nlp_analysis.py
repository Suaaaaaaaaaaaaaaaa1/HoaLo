import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

from underthesea import word_tokenize, sentiment
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora
from gensim.models import LdaModel

import re
from collections import Counter
import json
from datetime import datetime
import os

plt.rcParams['figure.figsize'] = (16, 8)
sns.set_palette("Set2")

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================================================================
# 1. LOAD DATA
# ===========================================================================
df = pd.read_csv(f'{OUTPUT_DIR}/posts_cleaned.csv', encoding='utf-8')
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df['date'] = pd.to_datetime(df['date'], errors='coerce')

print(f"Loaded {len(df):,} posts")
print(f"Posts with text: {df['has_text'].sum()}")
print(f"Avg text length: {df['text_length'].mean():.0f} chars")

df_text = df[df['has_text'] == True].copy()
print(f"\nAnalyzing {len(df_text):,} posts with text")

# ===========================================================================
# 2. VIETNAMESE STOPWORDS + PREPROCESSING
# ===========================================================================
VIETNAMESE_STOPWORDS = set([
    'và', 'của', 'có', 'được', 'trong', 'là', 'cho', 'với', 'từ', 'một',
    'các', 'này', 'những', 'đã', 'để', 'cũng', 'người', 'không', 'nhưng',
    'về', 'đến', 'hay', 'lại', 'thì', 'vào', 'ra', 'năm', 'ngày', 'tại',
    'như', 'theo', 'họ', 'nó', 'bởi', 'khi', 'nếu', 'đang', 'sẽ',
    'tôi', 'bạn', 'anh', 'chị', 'em', 'chúng_tôi', 'chúng_ta', 'mình',
    'ta', 'ông', 'bà', 'cô', 'chú', 'cậu', 'mày', 'tao',
    'trên', 'dưới', 'giữa', 'ngoài', 'sau', 'trước', 'bên', 'cùng',
    'qua', 'đối_với', 'nhờ', 'tới', 'khỏi',
    'mà', 'rằng', 'vì', 'nên', 'hoặc', 'hay_là', 'song', 'nhưng_mà',
    'do', 'bởi_vì', 'vì_vậy', 'tuy', 'dù',
    'gì', 'ai', 'đâu', 'nào', 'sao', 'thế_nào', 'bao_giờ', 'bao_nhiêu',
    'tại_sao', 'ở_đâu', 'ra_sao',
    'mỗi', 'mọi', 'cả', 'toàn', 'tất_cả', 'nhiều', 'ít', 'vài',
    'đủ', 'thêm', 'nữa', 'khác',
    'hôm_nay', 'ngày_mai', 'hôm_qua', 'bây_giờ', 'lúc', 'lúc_nào',
    'khi_nào', 'giờ', 'lúc_này', 'hiện_nay',
    'làm', 'đi', 'nói', 'thấy', 'biết', 'muốn', 'cần', 'phải',
    'bị', 'đưa', 'lấy', 'có_thể',
    'rất', 'quá', 'khá', 'hơn', 'nhất', 'chỉ', 'đều',
    'còn', 'vẫn', 'vừa', 'mới', 'luôn', 'thường',
    'hai', 'ba', 'bốn', 'sáu', 'bảy', 'tám', 'chín', 'mười',
    'cái', 'chiếc', 'con', 'quả', 'trái', 'bức', 'tờ', 'cuốn', 'quyển',
    'chưa', 'rồi'
])

print(f"✅ Loaded {len(VIETNAMESE_STOPWORDS)} Vietnamese stopwords")


def clean_text(text):
    if pd.isna(text) or text == '':
        return ''
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-ZÀ-ỹ\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_vietnamese(text):
    if not text:
        return []
    try:
        tokens = word_tokenize(text, format="text").split()
        tokens = [t for t in tokens if t not in VIETNAMESE_STOPWORDS]
        tokens = [t for t in tokens if len(t) > 1]
        return tokens
    except:
        return text.split()


print("Processing texts...")
df_text['text_clean'] = df_text['text'].apply(clean_text)
df_text['tokens'] = df_text['text_clean'].apply(tokenize_vietnamese)
df_text['token_count'] = df_text['tokens'].apply(len)

print(f"✅ Preprocessing complete")
print(f"Avg tokens per post: {df_text['token_count'].mean():.1f}")
print(f"Total unique tokens: {len(set([t for tokens in df_text['tokens'] for t in tokens])):,}")

# ===========================================================================
# 3. TOPIC MODELING (LDA)
# ===========================================================================
print("\nPreparing data for topic modeling...")

df_text['original_index'] = range(len(df_text))

min_tokens = 3
df_text['token_count_check'] = df_text['tokens'].apply(len)
valid_mask = df_text['token_count_check'] >= min_tokens

print(f"Posts with >= {min_tokens} tokens: {valid_mask.sum()}/{len(df_text)}")

texts = df_text[valid_mask]['tokens'].tolist()
valid_indices = df_text[valid_mask]['original_index'].tolist()

dictionary = corpora.Dictionary(texts)
print(f"Dictionary initial size: {len(dictionary)}")

dictionary.filter_extremes(no_below=2, no_above=0.8)
print(f"Dictionary after filtering: {len(dictionary)}")

corpus = [dictionary.doc2bow(text) for text in texts]

NUM_TOPICS = 7

print(f"Training LDA model with {NUM_TOPICS} topics...")
lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=NUM_TOPICS,
    random_state=42,
    passes=10,
    alpha='auto',
    per_word_topics=True
)

print(f"✅ LDA model trained")
print("\n" + "=" * 80)
print("DISCOVERED TOPICS")
print("=" * 80)

topic_labels = {}
for idx, topic in lda_model.print_topics(-1, num_words=10):
    print(f"\nTopic {idx}:")
    print(topic)
    words = [w.split('*')[1].replace('"', '').strip() for w in topic.split('+')]
    topic_labels[idx] = ', '.join(words[:5])


def get_dominant_topic(bow):
    if not bow or len(bow) == 0:
        return -1, 0.0
    topic_probs = lda_model.get_document_topics(bow)
    if not topic_probs or len(topic_probs) == 0:
        return -1, 0.0
    dominant = max(topic_probs, key=lambda x: x[1])
    return dominant[0], dominant[1]


df_text['topic_id'] = -1
df_text['topic_prob'] = 0.0

for i, (idx, bow) in enumerate(zip(valid_indices, corpus)):
    topic_id, prob = get_dominant_topic(bow)
    df_text.loc[df_text['original_index'] == idx, 'topic_id'] = topic_id
    df_text.loc[df_text['original_index'] == idx, 'topic_prob'] = prob

print("\n✅ Topics assigned")
print("\nTopic distribution:")
topic_dist = df_text['topic_id'].value_counts().sort_index()
print(topic_dist)

topic_neg1_count = (df_text['topic_id'] == -1).sum()
if topic_neg1_count > 0:
    print(f"\n⚠️  Topic -1: {topic_neg1_count} posts (too short)")

df_text['topic_label'] = df_text['topic_id'].apply(
    lambda x: topic_labels.get(x, 'Uncategorized') if x >= 0 else 'Too Short'
)

# --- Topic Visualization ---
df_topics_valid = df_text[df_text['topic_id'] >= 0].copy()

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

ax = axes[0, 0]
topic_counts = df_topics_valid['topic_id'].value_counts().sort_index()
colors = plt.cm.Set3(range(len(topic_counts)))
ax.bar(range(len(topic_counts)), topic_counts.values, color=colors)
ax.set_xlabel('Topic ID')
ax.set_ylabel('Number of Posts')
ax.set_title('Topic Distribution (Valid Topics Only)', fontweight='bold', fontsize=14)
ax.grid(axis='y', alpha=0.3)
for i, v in enumerate(topic_counts.values):
    ax.text(i, v, str(v), ha='center', va='bottom')

ax = axes[0, 1]
labels = [f"Topic {i}\n({topic_counts[i]} posts)" for i in topic_counts.index]
ax.pie(topic_counts.values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
ax.set_title('Topic Distribution (%)', fontweight='bold', fontsize=14)

ax = axes[1, 0]
topic_engagement = df_topics_valid.groupby('topic_id')['engagement_total'].mean().sort_index()
ax.bar(range(len(topic_engagement)), topic_engagement.values, color=colors)
ax.set_xlabel('Topic ID')
ax.set_ylabel('Avg Engagement')
ax.set_title('Average Engagement by Topic', fontweight='bold', fontsize=14)
ax.grid(axis='y', alpha=0.3)
for i, v in enumerate(topic_engagement.values):
    ax.text(i, v, f'{v:.0f}', ha='center', va='bottom')

ax = axes[1, 1]
df_topics_time = df_topics_valid[df_topics_valid['year_month'].notna()].copy()
if len(df_topics_time) > 0:
    topic_time = df_topics_time.groupby(['year_month', 'topic_id']).size().unstack(fill_value=0)
    topic_time.plot(kind='area', stacked=True, ax=ax, alpha=0.7, color=colors)
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Posts')
    ax.set_title('Topic Trends Over Time', fontweight='bold', fontsize=14)
    ax.legend(title='Topic', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.suptitle('TOPIC MODELING ANALYSIS', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/nlp_01_topic_modeling.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ nlp_01_topic_modeling.png")

# ===========================================================================
# 4. SENTIMENT ANALYSIS
# ===========================================================================
print("\nAnalyzing sentiment...")

POSITIVE_WORDS = set([
    'tuyệt', 'đẹp', 'hay', 'tốt', 'thích', 'yêu', 'xuất_sắc', 'ấn_tượng',
    'tuyệt_vời', 'hoàn_hảo', 'đáng', 'great', 'good', 'excellent',
    'amazing', 'wonderful', 'love', 'nice',
    'tự_hào', 'vinh_quang', 'anh_dũng', 'kiên_cường', 'vẻ_vang',
    'cảm_động', 'xúc_động', 'đáng_nhớ', 'phi_thường', 'vĩ_đại'
])

NEGATIVE_WORDS = set([
    'tệ', 'xấu', 'kém', 'không_tốt', 'thất_vọng', 'buồn', 'dở',
    'không_nên', 'bad', 'poor', 'terrible', 'disappointed', 'waste',
    'thiếu', 'nhàm', 'tẻ_nhạt', 'chán',
    'đau_thương', 'khổ_cực', 'tàn_bạo', 'đau_khổ', 'bi_thảm'
])


def sentiment_rule_based(tokens):
    if not tokens or len(tokens) == 0:
        return 'neutral'
    pos_count = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg_count = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    if pos_count > neg_count:
        return 'positive'
    elif neg_count > pos_count:
        return 'negative'
    else:
        return 'neutral'


df_text['sentiment'] = df_text['tokens'].apply(sentiment_rule_based)
df_text['sentiment_score'] = df_text['tokens'].apply(
    lambda tokens: sum(1 for t in tokens if t in POSITIVE_WORDS) -
                   sum(1 for t in tokens if t in NEGATIVE_WORDS)
)

print("✅ Sentiment assigned")
sentiment_dist = df_text['sentiment'].value_counts()
for sent, count in sentiment_dist.items():
    pct = count / len(df_text) * 100
    print(f"  {sent.capitalize()}: {count} ({pct:.1f}%)")

# --- Sentiment Visualization ---
colors_sent = {'positive': '#4CAF50', 'neutral': '#FFC107', 'negative': '#f44336'}

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

ax = axes[0, 0]
sentiment_counts = df_text['sentiment'].value_counts()
c = [colors_sent.get(s, 'gray') for s in sentiment_counts.index]
ax.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=c)
ax.set_title('Sentiment Distribution', fontweight='bold', fontsize=14)

ax = axes[0, 1]
df_text_time = df_text[df_text['year_month'].notna()].copy()
sent_time = df_text_time.groupby(['year_month', 'sentiment']).size().unstack(fill_value=0)
sent_time_pct = sent_time.div(sent_time.sum(axis=1), axis=0) * 100
sent_time_pct.plot(kind='bar', stacked=True, ax=ax, color=[colors_sent.get(col, 'gray') for col in sent_time_pct.columns])
ax.set_xlabel('Month')
ax.set_ylabel('Percentage (%)')
ax.set_title('Sentiment Trends Over Time', fontweight='bold', fontsize=14)
ax.legend(title='Sentiment')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)

ax = axes[1, 0]
sent_engagement = df_text.groupby('sentiment')['engagement_total'].mean()
c = [colors_sent.get(s, 'gray') for s in sent_engagement.index]
ax.bar(range(len(sent_engagement)), sent_engagement.values, color=c)
ax.set_xticks(range(len(sent_engagement)))
ax.set_xticklabels(sent_engagement.index)
ax.set_ylabel('Avg Engagement')
ax.set_title('Engagement by Sentiment', fontweight='bold', fontsize=14)
ax.grid(axis='y', alpha=0.3)

ax = axes[1, 1]
ax.hist(df_text['sentiment_score'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Neutral')
ax.set_xlabel('Sentiment Score')
ax.set_ylabel('Frequency')
ax.set_title('Sentiment Score Distribution', fontweight='bold', fontsize=14)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.suptitle('SENTIMENT ANALYSIS', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/nlp_02_sentiment_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ nlp_02_sentiment_analysis.png")

# ===========================================================================
# 5. KEYWORD ANALYSIS (TF-IDF + Word Frequency)
# ===========================================================================
print("\nExtracting keywords...")

corpus_text = df_text['tokens'].apply(lambda x: ' '.join(x)).tolist()

tfidf = TfidfVectorizer(
    max_features=100,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.7
)

tfidf_matrix = tfidf.fit_transform(corpus_text)
feature_names = tfidf.get_feature_names_out()

tfidf_scores = tfidf_matrix.sum(axis=0).A1
top_keywords_idx = tfidf_scores.argsort()[-30:][::-1]
top_keywords = [(feature_names[i], tfidf_scores[i]) for i in top_keywords_idx]

print("\n✅ Top 30 Keywords (TF-IDF):")
for i, (keyword, score) in enumerate(top_keywords, 1):
    print(f"{i:2}. {keyword:30} ({score:.2f})")

all_tokens = [token for tokens in df_text['tokens'] for token in tokens]
word_freq = Counter(all_tokens)
top_words = word_freq.most_common(50)

hoalo_keywords = {
    'tù': 0, 'giam': 0, 'tù_binh': 0, 'nhà_tù': 0,
    'lịch_sử': 0, 'cách_mạng': 0, 'chiến_sĩ': 0,
    'hiện_vật': 0, 'triển_lãm': 0, 'tham_quan': 0,
    'di_tích': 0, 'bảo_tàng': 0
}

for keyword in hoalo_keywords.keys():
    hoalo_keywords[keyword] = sum(1 for tokens in df_text['tokens'] if keyword in tokens)

# --- Keyword Visualization ---
fig, axes = plt.subplots(2, 2, figsize=(20, 14))

ax = axes[0, 0]
all_text = ' '.join(df_text['tokens'].apply(lambda x: ' '.join(x)))
wordcloud = WordCloud(width=800, height=400, background_color='white',
                      colormap='viridis', max_words=100).generate(all_text)
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
ax.set_title('Word Cloud - All Posts (Stopwords Removed)', fontweight='bold', fontsize=14)

ax = axes[0, 1]
top_20_words = word_freq.most_common(20)
words = [w[0] for w in top_20_words]
counts = [w[1] for w in top_20_words]
ax.barh(range(len(words)), counts, color='steelblue', alpha=0.7)
ax.set_yticks(range(len(words)))
ax.set_yticklabels(words)
ax.set_xlabel('Frequency')
ax.set_title('Top 20 Most Frequent Words', fontweight='bold', fontsize=14)
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()

ax = axes[1, 0]
top_15_tfidf = top_keywords[:15]
kw_words = [k[0] for k in top_15_tfidf]
kw_scores = [k[1] for k in top_15_tfidf]
ax.barh(range(len(kw_words)), kw_scores, color='coral', alpha=0.7)
ax.set_yticks(range(len(kw_words)))
ax.set_yticklabels(kw_words)
ax.set_xlabel('TF-IDF Score')
ax.set_title('Top 15 Keywords (TF-IDF)', fontweight='bold', fontsize=14)
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()

ax = axes[1, 1]
hoalo_items = [(k, v) for k, v in hoalo_keywords.items() if v > 0]
hoalo_items_sorted = sorted(hoalo_items, key=lambda x: x[1], reverse=True)
if hoalo_items_sorted:
    hl_words = [item[0] for item in hoalo_items_sorted]
    hl_counts = [item[1] for item in hoalo_items_sorted]
    ax.bar(range(len(hl_words)), hl_counts, color='purple', alpha=0.7)
    ax.set_xticks(range(len(hl_words)))
    ax.set_xticklabels(hl_words, rotation=45, ha='right')
    ax.set_ylabel('Number of Posts')
    ax.set_title('Hoa Lo Specific Keywords', fontweight='bold', fontsize=14)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('KEYWORD ANALYSIS', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/nlp_03_keyword_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ nlp_03_keyword_analysis.png")

# ===========================================================================
# 6. CROSS ANALYSIS
# ===========================================================================
print("\nCROSS ANALYSIS: Topics × Sentiment")
print("=" * 80)

df_analysis = df_text[df_text['topic_id'] >= 0].copy()
print(f"Analyzing {len(df_analysis)} posts with valid topics")

topic_sentiment = pd.crosstab(df_analysis['topic_id'], df_analysis['sentiment'], normalize='index') * 100
print("\nSentiment distribution by topic (%):")
print(topic_sentiment.round(1))

topic_stats = df_analysis.groupby('topic_id').agg({
    'engagement_total': ['mean', 'median', 'max'],
    'reactions_total': 'mean',
    'comment_count': 'mean',
    'share_count': 'mean',
    'postId': 'count'
}).round(1)

topic_stats.columns = ['_'.join(col).strip() for col in topic_stats.columns.values]
topic_stats = topic_stats.rename(columns={'postId_count': 'post_count'})

best_topic = topic_stats['engagement_total_mean'].idxmax()
print(f"\n✨ Best performing topic: Topic {best_topic}")
print(f"   Keywords: {topic_labels.get(best_topic, 'N/A')}")
print(f"   Avg engagement: {topic_stats.loc[best_topic, 'engagement_total_mean']:.1f}")

# --- Comprehensive Insights ---
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

ax1 = fig.add_subplot(gs[0, :])
topic_sent_count = pd.crosstab(df_text['topic_id'], df_text['sentiment'])
sns.heatmap(topic_sent_count, annot=True, fmt='d', cmap='YlGnBu', ax=ax1)
ax1.set_title('Topic × Sentiment Distribution', fontweight='bold', fontsize=14)
ax1.set_xlabel('Sentiment')
ax1.set_ylabel('Topic ID')

ax2 = fig.add_subplot(gs[1, 0])
topic_eng = df_text.groupby('topic_id')['engagement_total'].mean().sort_values(ascending=False)
ax2.barh(range(len(topic_eng)), topic_eng.values, color='green', alpha=0.7)
ax2.set_yticks(range(len(topic_eng)))
ax2.set_yticklabels([f"Topic {i}" for i in topic_eng.index])
ax2.set_xlabel('Avg Engagement')
ax2.set_title('Avg Engagement by Topic', fontweight='bold', fontsize=12)
ax2.grid(axis='x', alpha=0.3)

ax3 = fig.add_subplot(gs[1, 1])
topic_counts_all = df_text['topic_id'].value_counts().sort_index()
ax3.bar(range(len(topic_counts_all)), topic_counts_all.values, color='steelblue', alpha=0.7)
ax3.set_xticks(range(len(topic_counts_all)))
ax3.set_xticklabels([f"T{i}" for i in topic_counts_all.index])
ax3.set_ylabel('Number of Posts')
ax3.set_title('Posts Distribution by Topic', fontweight='bold', fontsize=12)
ax3.grid(axis='y', alpha=0.3)

ax4 = fig.add_subplot(gs[1, 2])
topic_sent_pct = pd.crosstab(df_text['topic_id'], df_text['sentiment'], normalize='index') * 100
topic_sent_pct.plot(kind='bar', stacked=True, ax=ax4,
                    color=[colors_sent.get(col, 'gray') for col in topic_sent_pct.columns])
ax4.set_xlabel('Topic ID')
ax4.set_ylabel('Percentage (%)')
ax4.set_title('Sentiment % by Topic', fontweight='bold', fontsize=12)
ax4.legend(title='Sentiment', loc='upper right')
ax4.grid(axis='y', alpha=0.3)
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=0)

ax5 = fig.add_subplot(gs[2, 0])
topic_eng_sorted = df_text[df_text['topic_id'] >= 0].groupby('topic_id')['engagement_total'].mean().sort_values(ascending=True)
ax5.barh(range(len(topic_eng_sorted)), topic_eng_sorted.values,
         color=plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(topic_eng_sorted))))
ax5.set_yticks(range(len(topic_eng_sorted)))
ax5.set_yticklabels([f'Topic {i}' for i in topic_eng_sorted.index])
ax5.set_xlabel('Avg Engagement')
ax5.set_title('Topic Performance Ranking', fontweight='bold', fontsize=12)
ax5.grid(axis='x', alpha=0.3)

ax6 = fig.add_subplot(gs[2, 1])
ax6.scatter(df_text['text_length'], df_text['engagement_total'], alpha=0.3, s=20)
ax6.set_xlabel('Text Length (chars)')
ax6.set_ylabel('Engagement')
ax6.set_title('Engagement vs Text Length', fontweight='bold', fontsize=12)
ax6.grid(alpha=0.3)
corr = df_text[['text_length', 'engagement_total']].corr().iloc[0, 1]
ax6.text(0.05, 0.95, f'Corr: {corr:.3f}', transform=ax6.transAxes,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax7 = fig.add_subplot(gs[2, 2])
sent_eng = df_text.groupby('sentiment')['engagement_total'].mean()
c = [colors_sent.get(s, 'gray') for s in sent_eng.index]
ax7.bar(range(len(sent_eng)), sent_eng.values, color=c)
ax7.set_xticks(range(len(sent_eng)))
ax7.set_xticklabels(sent_eng.index)
ax7.set_ylabel('Avg Engagement')
ax7.set_title('Avg Engagement by Sentiment', fontweight='bold', fontsize=12)
ax7.grid(axis='y', alpha=0.3)

plt.suptitle('COMPREHENSIVE NLP INSIGHTS', fontsize=18, fontweight='bold')
plt.savefig(f'{OUTPUT_DIR}/nlp_04_comprehensive_insights.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ nlp_04_comprehensive_insights.png")

# ===========================================================================
# 7. SAVE OUTPUTS
# ===========================================================================
df_text_export = df_text[[
    'postId', 'date', 'text', 'text_length',
    'topic_id', 'topic_label', 'topic_prob',
    'sentiment', 'sentiment_score',
    'engagement_total', 'reactions_total', 'comment_count', 'share_count'
]].copy()

df_text_export.to_csv(f'{OUTPUT_DIR}/posts_nlp_enriched.csv', index=False, encoding='utf-8-sig')
print(f"\n✅ Saved: posts_nlp_enriched.csv")

nlp_report = {
    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset': {
        'total_posts': int(len(df_text)),
        'avg_text_length': float(df_text['text_length'].mean()),
        'total_tokens': int(df_text['token_count'].sum())
    },
    'topics': {
        'num_topics': NUM_TOPICS,
        'topic_distribution': topic_counts.to_dict(),
        'topic_labels': topic_labels,
        'best_topic': int(best_topic),
        'best_topic_engagement': float(topic_stats.loc[best_topic, 'engagement_total_mean'])
    },
    'sentiment': {
        'distribution': sentiment_dist.to_dict(),
        'positive_pct': float(sentiment_dist.get('positive', 0) / len(df_text) * 100),
        'neutral_pct': float(sentiment_dist.get('neutral', 0) / len(df_text) * 100),
        'negative_pct': float(sentiment_dist.get('negative', 0) / len(df_text) * 100)
    },
    'keywords': {
        'top_20_words': [(w, int(c)) for w, c in top_words[:20]],
        'top_15_tfidf': [(k, float(s)) for k, s in top_keywords[:15]],
        'hoalo_keywords': {k: int(v) for k, v in hoalo_keywords.items() if v > 0}
    },
    'insights': {
        'engagement_sentiment_corr': float(df_text[['sentiment_score', 'engagement_total']].corr().iloc[0, 1]),
        'engagement_length_corr': float(corr),
        'best_sentiment': str(sent_eng.idxmax()),
        'best_sentiment_engagement': float(sent_eng.max())
    }
}

with open(f'{OUTPUT_DIR}/nlp_analysis_report.json', 'w', encoding='utf-8') as f:
    json.dump(nlp_report, f, indent=2, ensure_ascii=False)

print("✅ Saved: nlp_analysis_report.json")

print("\n" + "=" * 80)
print("NLP ANALYSIS COMPLETE!")
print("=" * 80)
print(f"  Posts analyzed: {len(df_text):,}")
print(f"  Topics: {NUM_TOPICS} | Best: Topic {best_topic}")
print(f"  Sentiment: {sentiment_dist.get('positive', 0)} pos / {sentiment_dist.get('neutral', 0)} neu / {sentiment_dist.get('negative', 0)} neg")
print(f"  Top keyword: {top_words[0][0]} ({top_words[0][1]} times)")
