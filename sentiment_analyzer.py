import streamlit as st
import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# =========================
# UI
# =========================
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🛍️")
st.title("🛍️ Amazon Review Sentiment Analyzer")
st.write("Enter a review and get sentiment prediction")

# =========================
# DATASET (IMPROVED)
# =========================
positive = [
    "This product is amazing", "I love this item", "Best purchase ever",
    "Excellent quality", "Very happy with this", "Highly recommended",
    "Works perfectly", "Great value for money", "Superb experience",
    "Fantastic product"
]

negative = [
    "Worst product ever", "I hate it", "Very bad quality",
    "Completely useless", "Broke after one day", "Waste of money",
    "Terrible experience", "Do not buy this", "Very disappointed",
    "Poor quality product"
]

neutral = [
    "It is okay", "Average product", "Not bad not good",
    "It works fine", "Normal item", "Nothing special",
    "Decent enough", "It's fine", "Basic product", "Alright item"
]

data = (
    [(r, "Positive") for r in positive] +
    [(r, "Negative") for r in negative] +
    [(r, "Neutral") for r in neutral]
)

df = pd.DataFrame(data, columns=["review", "sentiment"])

# =========================
# CLEAN TEXT (NO NLTK)
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df["clean"] = df["review"].apply(clean_text)

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"],
    df["sentiment"],
    test_size=0.25,
    random_state=42,
    stratify=df["sentiment"]
)

# =========================
# MODEL (IMPROVED)
# =========================
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
])

model.fit(X_train, y_train)

# accuracy
acc = accuracy_score(y_test, model.predict(X_test))

st.sidebar.write("📊 Model Accuracy")
st.sidebar.write(f"{acc*100:.2f}%")

# =========================
# INPUT
# =========================
user_input = st.text_area("Enter your review")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter text")
    else:
        cleaned = clean_text(user_input)
        pred = model.predict([cleaned])[0]

        emoji = {
            "Positive": "😊",
            "Negative": "😡",
            "Neutral": "😐"
        }

        st.success(f"Prediction: {emoji[pred]} {pred}")
