"""
============================================================
  AMAZON REVIEW SENTIMENT ANALYZER
  Text Mining Project — Undergraduate Level
  Run: python sentiment_analyzer.py
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score, roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import warnings
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  STEP 0: Download NLTK resources
# ─────────────────────────────────────────────
print("\n📦 Downloading NLTK resources...")
for resource in ["stopwords", "wordnet", "punkt", "punkt_tab", "averaged_perceptron_tagger"]:
    nltk.download(resource, quiet=True)

# ─────────────────────────────────────────────
#  STEP 1: Generate Synthetic Amazon-Style Dataset
# ─────────────────────────────────────────────
print("📊 Generating Amazon-style review dataset...")

np.random.seed(42)

positive_reviews = [
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

negative_reviews = [
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

neutral_reviews = [
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

# Build DataFrame
all_reviews = (
    [(r, "Positive") for r in positive_reviews] +
    [(r, "Negative") for r in negative_reviews] +
    [(r, "Neutral")  for r in neutral_reviews]
)

# Augment with slight variations
augmented = []
for review, label in all_reviews:
    augmented.append((review, label))
    words = review.split()
    if len(words) > 5:
        shuffled = words[:2] + words[2:]
        augmented.append((" ".join(shuffled), label))

df = pd.DataFrame(augmented, columns=["review", "sentiment"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"   ✅ Dataset created: {len(df)} reviews")
print(f"   Distribution:\n{df['sentiment'].value_counts().to_string()}\n")

# ─────────────────────────────────────────────
#  STEP 2: Text Preprocessing
# ─────────────────────────────────────────────
print("🧹 Preprocessing text...")

lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words("english"))

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)        # Remove URLs
    text = re.sub(r"[^a-z\s]", "", text)              # Keep letters only
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

df["clean_review"] = df["review"].apply(preprocess)
print("   ✅ Preprocessing done!\n")

# ─────────────────────────────────────────────
#  STEP 3: Feature Engineering — TF-IDF
# ─────────────────────────────────────────────
X = df["clean_review"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────────
#  STEP 4: Train Multiple Models
# ─────────────────────────────────────────────
print("🤖 Training models...\n")

models = {
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ("clf",   LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "Naive Bayes": Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ("clf",   MultinomialNB())
    ]),
    "Linear SVM": Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ("clf",   LinearSVC(random_state=42, max_iter=2000))
    ]),
}

results = {}
for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    results[name] = {"pipeline": pipeline, "y_pred": y_pred, "accuracy": acc}
    print(f"   ✅ {name:25s} → Accuracy: {acc*100:.2f}%")

best_name     = max(results, key=lambda k: results[k]["accuracy"])
best_pipeline = results[best_name]["pipeline"]
print(f"\n🏆 Best Model: {best_name} ({results[best_name]['accuracy']*100:.2f}%)\n")

# ─────────────────────────────────────────────
#  STEP 5: Detailed Report for Best Model
# ─────────────────────────────────────────────
y_pred_best = results[best_name]["y_pred"]
print("=" * 55)
print(f"  Classification Report — {best_name}")
print("=" * 55)
print(classification_report(y_test, y_pred_best))

# ─────────────────────────────────────────────
#  STEP 6: Visualizations
# ─────────────────────────────────────────────
print("📈 Generating visualizations...")

os.makedirs("plots", exist_ok=True)

COLORS = {
    "Positive": "#2ecc71",
    "Negative": "#e74c3c",
    "Neutral":  "#3498db",
}
PALETTE = ["#2ecc71", "#e74c3c", "#3498db"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# --- Plot 1: Sentiment Distribution ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Sentiment Distribution", fontsize=16, fontweight="bold", y=1.02)

counts = df["sentiment"].value_counts()
bars   = axes[0].bar(counts.index, counts.values,
                     color=[COLORS[s] for s in counts.index],
                     edgecolor="white", linewidth=1.5, width=0.55)
for bar, val in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 str(val), ha="center", va="bottom", fontweight="bold")
axes[0].set_title("Review Count by Sentiment")
axes[0].set_xlabel("Sentiment")
axes[0].set_ylabel("Count")

axes[1].pie(counts.values, labels=counts.index,
            colors=[COLORS[s] for s in counts.index],
            autopct="%1.1f%%", startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[1].set_title("Sentiment Share (%)")

plt.tight_layout()
plt.savefig("plots/01_sentiment_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# --- Plot 2: Model Accuracy Comparison ---
fig, ax = plt.subplots(figsize=(9, 5))
names  = list(results.keys())
accs   = [results[n]["accuracy"] * 100 for n in names]
bar_colors = ["#e74c3c" if n == best_name else "#95a5a6" for n in names]

bars = ax.bar(names, accs, color=bar_colors, edgecolor="white",
              linewidth=1.5, width=0.5)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{acc:.2f}%", ha="center", va="bottom", fontweight="bold")
ax.set_ylim(0, 115)
ax.set_title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
ax.set_ylabel("Accuracy (%)")
ax.set_xlabel("Model")
best_patch = mpatches.Patch(color="#e74c3c", label=f"Best: {best_name}")
ax.legend(handles=[best_patch], loc="upper right")
plt.tight_layout()
plt.savefig("plots/02_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# --- Plot 3: Confusion Matrix ---
labels = ["Negative", "Neutral", "Positive"]
cm     = confusion_matrix(y_test, y_pred_best, labels=labels)

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
            xticklabels=labels, yticklabels=labels,
            linewidths=0.5, linecolor="white", ax=ax)
ax.set_title(f"Confusion Matrix — {best_name}", fontsize=13, fontweight="bold")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
plt.tight_layout()
plt.savefig("plots/03_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

# --- Plot 4: Review Length Distribution ---
df["review_length"] = df["review"].apply(lambda x: len(x.split()))

fig, ax = plt.subplots(figsize=(10, 5))
for sentiment, color in COLORS.items():
    subset = df[df["sentiment"] == sentiment]["review_length"]
    ax.hist(subset, bins=20, alpha=0.6, label=sentiment,
            color=color, edgecolor="white")
ax.set_title("Review Word-Length Distribution by Sentiment",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Word Count")
ax.set_ylabel("Frequency")
ax.legend()
plt.tight_layout()
plt.savefig("plots/04_length_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# --- Plot 5: Top TF-IDF Features ---
tfidf_vect  = best_pipeline.named_steps["tfidf"]
clf         = best_pipeline.named_steps["clf"]
feature_names = np.array(tfidf_vect.get_feature_names_out())

if hasattr(clf, "coef_"):
    classes = clf.classes_
    fig, axes = plt.subplots(1, len(classes), figsize=(15, 5))
    fig.suptitle(f"Top Predictive Words — {best_name}",
                 fontsize=14, fontweight="bold")
    for i, (cls, ax) in enumerate(zip(classes, axes)):
        top_idx  = np.argsort(clf.coef_[i])[-15:]
        top_feat = feature_names[top_idx]
        top_coef = clf.coef_[i][top_idx]
        ax.barh(top_feat, top_coef, color=COLORS.get(cls, "#7f8c8d"),
                edgecolor="white")
        ax.set_title(cls, fontweight="bold", color=COLORS.get(cls, "#7f8c8d"))
        ax.set_xlabel("TF-IDF Coefficient")
    plt.tight_layout()
    plt.savefig("plots/05_top_features.png", dpi=150, bbox_inches="tight")
    plt.close()

print("   ✅ All plots saved to  ./plots/\n")

# ─────────────────────────────────────────────
#  STEP 7: Live Prediction Demo
# ─────────────────────────────────────────────
print("=" * 55)
print("  🎯  LIVE PREDICTION DEMO")
print("=" * 55)

demo_reviews = [
    "This product is absolutely wonderful, I love it!",
    "Terrible quality, broke after one day. Total waste.",
    "It's okay, nothing special but works fine.",
    "Best thing I ever bought. Highly recommended!",
    "Very disappointing. Does not work as advertised.",
]

emoji_map = {"Positive": "😊", "Negative": "😡", "Neutral": "😐"}

for review in demo_reviews:
    prediction = best_pipeline.predict([review])[0]
    print(f"  Review   : {review[:60]}...")
    print(f"  Predicted: {emoji_map[prediction]} {prediction}\n")

# ─────────────────────────────────────────────
#  STEP 8: Interactive Prediction
# ─────────────────────────────────────────────
print("=" * 55)
print("  ✍️  TYPE YOUR OWN REVIEW  (type 'quit' to exit)")
print("=" * 55)

while True:
    user_input = input("\n  Enter a review: ").strip()
    if user_input.lower() in ("quit", "exit", "q"):
        print("\n  👋 Exiting. Thanks for using the Sentiment Analyzer!\n")
        break
    if not user_input:
        continue
    pred = best_pipeline.predict([user_input])[0]
    print(f"  🔍 Sentiment: {emoji_map[pred]} {pred}")
