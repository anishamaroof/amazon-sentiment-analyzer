
#   AMAZON REVIEW SENTIMENT Analyzer 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from wordcloud import WordCloud
import re
import nltk
import warnings
warnings.filterwarnings("ignore")

nltk.download('stopwords',                  quiet=True)
nltk.download('wordnet',                    quiet=True)
nltk.download('punkt',                      quiet=True)
nltk.download('punkt_tab',                  quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from collections import Counter

print("✅ All libraries loaded successfully!")

# ════════════════════════════════════════════════════════════
#  CELL 2 — RAW DATASET
# ════════════════════════════════════════════════════════════
positive = [
    "This product is absolutely amazing, exceeded all my expectations!",
    "Best purchase I have ever made, highly recommend to everyone.",
    "Outstanding quality and fast shipping. Will definitely buy again.",
    "Fantastic product, works perfectly and looks great.",
    "I love this item so much, it changed my daily routine for the better.",
    "Superb quality, worth every penny. Very happy with this purchase.",
    "Great value for money. Product arrived on time and works flawlessly.",
    "Excellent product! The quality is top notch and delivery was quick.",
    "Five stars! Exactly as described and works better than expected.",
    "Perfect product! My family loves it. Would definitely recommend.",
    "Incredible quality for the price. Packaging was great too.",
    "So happy with this purchase! Arrived quickly and works perfectly.",
    "Really good product, highly recommend. Seller was very responsive.",
    "Amazing value! Sturdy, well-made, and looks exactly like the pictures.",
    "This is the best product in its category. Absolutely love it!",
    "Wonderfully crafted and very durable. Worth the price.",
    "Everything about this product is perfect. Great customer service too.",
    "Very satisfied with the purchase. Product quality is exceptional.",
    "Love love love this product! Will be buying more as gifts.",
    "Top quality item! Fast shipping, well packaged, and works great.",
    "Brilliant product! Exactly what I needed. Very easy to use.",
    "Highly satisfied! This product is genuinely worth the investment.",
    "Awesome product! Well built and easy to set up. No complaints at all.",
    "Extremely happy with this buy. Delivery was faster than expected.",
    "This item is perfect for daily use. Strongly recommend it!",
]

negative = [
    "Terrible product, broke after just two days of use. Very disappointed.",
    "Worst purchase ever. Completely useless and waste of money.",
    "Do not buy this! It stopped working after one week. Poor quality.",
    "Very disappointed. Product looks nothing like the pictures online.",
    "Awful experience. Customer service was unhelpful and rude.",
    "This product is a scam. Fake reviews. Do not trust.",
    "Broke immediately. Absolute garbage. I want my money back.",
    "Poor quality materials. Fell apart within days. Not worth it at all.",
    "Terrible item! Defective out of the box. Very frustrating experience.",
    "Would give zero stars if I could. This product is completely useless.",
    "Extremely disappointing. Nothing works as advertised. Avoid this seller.",
    "Waste of money! Product arrived damaged and packaging was terrible.",
    "Horrible product. Cheap material, bad smell, and it doesn't work.",
    "Never buying from this seller again. Product is totally defective.",
    "This is junk. Fell apart on first use. Returning immediately.",
    "Very bad quality. Not as described. The seller is dishonest.",
    "Regret buying this. It overheated and stopped working after a day.",
    "Product is faulty and dangerous. Do not buy!",
    "Extremely poor build quality. Looks cheap and feels even cheaper.",
    "Total disappointment. The product malfunctioned right out of the box.",
    "Broken on arrival. Seller refused to refund. Disgusting service.",
    "Don't waste your money. This product is absolute garbage.",
    "Frustrating experience from start to finish. Product is defective.",
    "Terrible design and build. Feels like it was made to fail quickly.",
    "Completely useless product. Would not recommend to my worst enemy.",
]

neutral = [
    "Product is okay. Nothing special but it gets the job done.",
    "Average quality for the price. Expected better but not the worst.",
    "It works as described. Nothing more, nothing less. Decent purchase.",
    "Mediocre product. Some features work, others don't. Mixed feelings.",
    "Fairly standard item. Does what it's supposed to do.",
    "Not great, not terrible. Just an average product overall.",
    "Decent quality but I've seen better at this price range.",
    "It's alright. Meets basic expectations but lacks premium feel.",
    "Works fine for basic use. Wouldn't call it impressive though.",
    "Average experience. Shipping was on time but product is just ok.",
    "It's a basic product. Does the job but nothing extraordinary.",
    "Moderate quality. Acceptable for the price but don't expect much.",
    "Product is as described. Regular quality, nothing to rave about.",
    "It serves its purpose adequately. Neither impressed nor disappointed.",
    "Middle of the road product. Has its pros and cons.",
]

data = ([(r, "Positive") for r in positive] +
        [(r, "Negative") for r in negative] +
        [(r, "Neutral")  for r in neutral])

df = pd.DataFrame(data, columns=["review", "sentiment"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"✅ Dataset created: {len(df)} reviews")
print(df["sentiment"].value_counts())
print("\n📄 Sample reviews:")
print(df[["review","sentiment"]].head(6).to_string(index=False))

# ════════════════════════════════════════════════════════════
#  CELL 3 — TEXT MINING PIPELINE (Step by Step)
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("   📌 TEXT MINING PIPELINE")
print("="*60)

stemmer    = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# ── Step 1: Lowercasing ──────────────────────────────────────
def step1_lowercase(text):
    return text.lower()

# ── Step 2: Noise Removal ────────────────────────────────────
def step2_remove_noise(text):
    text = re.sub(r"http\S+|www\S+", "", text)   # remove URLs
    text = re.sub(r"[^a-z\s]",       "", text)   # remove punctuation/numbers
    text = re.sub(r"\s+",            " ", text)  # remove extra spaces
    return text.strip()

# ── Step 3: Tokenization ─────────────────────────────────────
def step3_tokenize(text):
    return word_tokenize(text)

# ── Step 4: Stopword Removal ─────────────────────────────────
def step4_remove_stopwords(tokens):
    return [t for t in tokens if t not in stop_words and len(t) > 2]

# ── Step 5: POS Tagging ──────────────────────────────────────
def step5_pos_tag(tokens):
    return pos_tag(tokens)

# ── Step 6: Stemming ─────────────────────────────────────────
def step6_stem(tokens):
    return [stemmer.stem(t) for t in tokens]

# ── Step 7: Lemmatization ────────────────────────────────────
def step7_lemmatize(tokens):
    return [lemmatizer.lemmatize(t) for t in tokens]

# ── Full Pipeline ────────────────────────────────────────────
def full_pipeline(text, return_stages=False):
    s1 = step1_lowercase(text)
    s2 = step2_remove_noise(s1)
    s3 = step3_tokenize(s2)
    s4 = step4_remove_stopwords(s3)
    s6 = step6_stem(s4)
    s7 = step7_lemmatize(s4)

    if return_stages:
        return {
            "1_lowercase":      s1,
            "2_noise_removed":  s2,
            "3_tokens":         s3,
            "4_no_stopwords":   s4,
            "5_pos_tags":       step5_pos_tag(s4),
            "6_stemmed":        s6,
            "7_lemmatized":     s7,
            "final_clean":      " ".join(s7),
        }
    return " ".join(s7)

# ── Show Pipeline on 1 example ───────────────────────────────
example = df["review"].iloc[0]
stages  = full_pipeline(example, return_stages=True)

print(f"\n🔍 Example Review:\n   \"{example}\"\n")
print("─── PIPELINE STAGES ───────────────────────────────────")
for stage, result in stages.items():
    if stage == "5_pos_tags":
        print(f"  {stage:20s}: {result[:5]} ...")
    elif isinstance(result, list):
        print(f"  {stage:20s}: {result[:8]} ...")
    else:
        print(f"  {stage:20s}: {str(result)[:80]}")

# ── Apply to whole dataset ───────────────────────────────────
df["clean"] = df["review"].apply(full_pipeline)
print("\n✅ Pipeline applied to all reviews!")
print(df[["review","clean","sentiment"]].head(3).to_string(index=False))

# ════════════════════════════════════════════════════════════
#  CELL 4 — FEATURE EXTRACTION (BoW + TF-IDF)
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("   📌 FEATURE EXTRACTION")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["sentiment"],
    test_size=0.2, random_state=42, stratify=df["sentiment"]
)

# Bag of Words
bow_vec = CountVectorizer(max_features=500)
X_bow   = bow_vec.fit_transform(df["clean"])
print(f"\n✅ Bag of Words matrix shape  : {X_bow.shape}")

# TF-IDF
tfidf_vec = TfidfVectorizer(ngram_range=(1,2), max_features=3000)
X_tfidf   = tfidf_vec.fit_transform(df["clean"])
print(f"✅ TF-IDF matrix shape        : {X_tfidf.shape}")

# Top BoW terms
top_bow = sorted(zip(bow_vec.get_feature_names_out(),
                     np.asarray(X_bow.sum(axis=0)).flatten()),
                 key=lambda x: -x[1])[:10]
print("\n📊 Top 10 Bag-of-Words terms:")
for word, freq in top_bow:
    print(f"   {word:20s} → {int(freq)}")

# ════════════════════════════════════════════════════════════
#  CELL 5 — TRAIN 3 ML MODELS
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("   📌 MODEL TRAINING")
print("="*60)

models = {
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=3000)),
        ("clf",   LogisticRegression(max_iter=1000))
    ]),
    "Naive Bayes": Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=3000)),
        ("clf",   MultinomialNB())
    ]),
    "Linear SVM": Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=3000)),
        ("clf",   LinearSVC(max_iter=2000))
    ]),
}

results = {}
print()
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred_m = pipe.predict(X_test)
    acc      = accuracy_score(y_test, y_pred_m)
    results[name] = {"pipe": pipe, "acc": acc, "y_pred": y_pred_m}
    print(f"  ✅ {name:25s} → Accuracy: {acc*100:.2f}%")

best      = max(results, key=lambda k: results[k]["acc"])
best_pipe = results[best]["pipe"]
y_pred    = results[best]["y_pred"]
print(f"\n  🏆 Best Model: {best} ({results[best]['acc']*100:.2f}%)")

# ════════════════════════════════════════════════════════════
#  CELL 6 — CLASSIFICATION REPORT
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print(f"   📋 CLASSIFICATION REPORT — {best}")
print("="*60)
print(classification_report(y_test, y_pred))

# ════════════════════════════════════════════════════════════
#  CELL 7 — VISUALIZATIONS (6 Charts)
# ════════════════════════════════════════════════════════════
COLORS = {"Positive":"#2ecc71", "Negative":"#e74c3c", "Neutral":"#3498db"}

# Chart 1 & 2: Distribution + Model Accuracy
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Dataset Overview & Model Performance", fontsize=14, fontweight="bold")

counts = df["sentiment"].value_counts()
axes[0].bar(counts.index, counts.values,
            color=[COLORS[s] for s in counts.index],
            edgecolor="white", width=0.5)
for bar, val in zip(axes[0].patches, counts.values):
    axes[0].text(bar.get_x()+bar.get_width()/2,
                 bar.get_height()+0.3, str(val), ha="center", fontweight="bold")
axes[0].set_title("Sentiment Distribution"); axes[0].set_ylabel("Count")

names_ = list(results.keys())
accs_  = [results[n]["acc"]*100 for n in names_]
bc     = ["#e74c3c" if n==best else "#bdc3c7" for n in names_]
axes[1].bar(names_, accs_, color=bc, edgecolor="white", width=0.5)
axes[1].set_ylim(0, 115)
axes[1].set_title("Model Accuracy Comparison"); axes[1].set_ylabel("Accuracy (%)")
for i,(n,a) in enumerate(zip(names_,accs_)):
    axes[1].text(i, a+1, f"{a:.1f}%", ha="center", fontweight="bold")
axes[1].legend(handles=[mpatches.Patch(color="#e74c3c", label=f"Best: {best}")])
plt.tight_layout(); plt.show()

# Chart 3: Confusion Matrix
labels_ = ["Negative","Neutral","Positive"]
cm = confusion_matrix(y_test, y_pred, labels=labels_)
fig, ax = plt.subplots(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
            xticklabels=labels_, yticklabels=labels_,
            linewidths=0.5, linecolor="white", ax=ax)
ax.set_title(f"Confusion Matrix — {best}", fontsize=13, fontweight="bold")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout(); plt.show()

# Chart 4: Word Clouds
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Word Clouds by Sentiment", fontsize=14, fontweight="bold")
wc_colors = {"Positive":"Greens", "Negative":"Reds", "Neutral":"Blues"}
for ax, sentiment in zip(axes, ["Positive","Negative","Neutral"]):
    text = " ".join(df[df["sentiment"]==sentiment]["clean"])
    wc   = WordCloud(width=500, height=300, background_color="white",
                     colormap=wc_colors[sentiment], max_words=60).generate(text)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(sentiment, fontsize=13, fontweight="bold", color=COLORS[sentiment])
plt.tight_layout(); plt.show()

# Chart 5: Top Words per Sentiment
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Top 10 Most Frequent Words per Sentiment", fontsize=14, fontweight="bold")
for ax, sentiment in zip(axes, ["Positive","Negative","Neutral"]):
    words  = " ".join(df[df["sentiment"]==sentiment]["clean"]).split()
    common = Counter(words).most_common(10)
    w, c   = zip(*common)
    ax.barh(w[::-1], c[::-1], color=COLORS[sentiment], edgecolor="white")
    ax.set_title(sentiment, fontweight="bold", color=COLORS[sentiment])
    ax.set_xlabel("Frequency")
plt.tight_layout(); plt.show()

# Chart 6: Pipeline Token Reduction
df["raw_len"]   = df["review"].apply(lambda x: len(x.split()))
df["clean_len"] = df["clean"].apply(lambda x: len(x.split()))
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(df))
ax.plot(x, df["raw_len"],   alpha=0.5, label="Before Pipeline", color="#95a5a6")
ax.plot(x, df["clean_len"], alpha=0.8, label="After Pipeline",  color="#e74c3c")
ax.fill_between(x, df["clean_len"], df["raw_len"], alpha=0.1, color="#e74c3c")
ax.set_title("Text Mining Pipeline — Token Reduction Effect", fontsize=13, fontweight="bold")
ax.set_xlabel("Review Index"); ax.set_ylabel("Word Count")
ax.legend(); plt.tight_layout(); plt.show()

print("✅ All 6 visualizations done!")

# ════════════════════════════════════════════════════════════
#  CELL 8 — LIVE DEMO PREDICTIONS
# ════════════════════════════════════════════════════════════
emoji = {"Positive":"😊", "Negative":"😡", "Neutral":"😐"}

demo = [
    "This product is absolutely wonderful, I love it!",
    "Terrible quality, broke after one day. Total waste of money.",
    "It is okay, nothing special but gets the job done.",
    "Best thing I ever bought. Highly recommended to everyone!",
    "Very disappointing. Does not work as advertised at all.",
]

print("\n" + "="*60)
print("   🎯 LIVE PREDICTIONS — Best Model:", best)
print("="*60)
for review in demo:
    cleaned = full_pipeline(review)
    pred    = best_pipe.predict([cleaned])[0]
    print(f"\n  Review   : {review}")
    print(f"  Cleaned  : {cleaned}")
    print(f"  Sentiment: {emoji[pred]} {pred}")

print("\n" + "="*60)
print("  ✅ PROJECT COMPLETE — PIPELINE SUMMARY")
print("="*60)
print("  Step 1: Lowercasing       → normalize text case")
print("  Step 2: Noise Removal     → strip URLs & punctuation")
print("  Step 3: Tokenization      → split into word tokens")
print("  Step 4: Stopword Removal  → remove common words")
print("  Step 5: POS Tagging       → identify word types")
print("  Step 6: Stemming          → reduce to root form")
print("  Step 7: Lemmatization     → proper dictionary form")
print("  Step 8: Feature Extraction→ Bag of Words + TF-IDF")
print("  Step 9: ML Classification → LR, Naive Bayes, SVM")
print("="*60)
