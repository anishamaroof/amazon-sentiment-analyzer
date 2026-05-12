import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import warnings

warnings.filterwarnings("ignore")

# ==============================
# Download NLTK Resources
# ==============================
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# ==============================
# Streamlit Page Config
# ==============================
st.set_page_config(
    page_title="Amazon Review Sentiment Analyzer",
    page_icon="🛍️",
    layout="centered"
)

# ==============================
# Title
# ==============================
st.title("🛍️ Amazon Review Sentiment Analyzer")
st.markdown("### Text Mining & Machine Learning Project")

st.write(
    """
This app predicts whether an Amazon review is:

- 😊 Positive
- 😐 Neutral
- 😡 Negative
"""
)

# ==============================
# Dataset
# ==============================
positive = [
    "This product is absolutely amazing, exceeded all my expectations!",
    "Best purchase I have ever made, highly recommend to everyone.",
    "Outstanding quality and fast shipping. Will definitely buy again.",
    "Fantastic product, works perfectly and looks great.",
    "I love this item so much, it changed my daily routine for the better.",
]

negative = [
    "Terrible product, broke after just two days of use. Very disappointed.",
    "Worst purchase ever. Completely useless and waste of money.",
    "Do not buy this! It stopped working after one week. Poor quality.",
    "Very disappointed. Product looks nothing like the pictures online.",
    "Awful experience. Customer service was unhelpful and rude.",
]

neutral = [
    "Product is okay. Nothing special but it gets the job done.",
    "Average quality for the price. Expected better but not the worst.",
    "It works as described. Nothing more, nothing less.",
    "Fairly standard item. Does what it's supposed to do.",
    "Not great, not terrible. Just an average product overall.",
]

data = (
    [(r, "Positive") for r in positive] +
    [(r, "Negative") for r in negative] +
    [(r, "Neutral") for r in neutral]
)

df = pd.DataFrame(data, columns=["review", "sentiment"])

# ==============================
# Text Cleaning
# ==============================
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove punctuation/numbers
    text = re.sub(r"[^a-z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenization
    tokens = text.split()

    # Stopword removal
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    # Lemmatization
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)

df["clean"] = df["review"].apply(clean_text)

# ==============================
# Train Model
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"],
    df["sentiment"],
    test_size=0.2,
    random_state=42
)

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression())
])

model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# ==============================
# Sidebar
# ==============================
st.sidebar.title("📊 Model Information")
st.sidebar.write(f"Model Accuracy: **{accuracy*100:.2f}%**")

st.sidebar.write("### Project Created By")
st.sidebar.write("""
- Anisha Maroof  

""")

# ==============================
# User Input
# ==============================
st.subheader("✍️ Enter Your Review")

user_review = st.text_area(
    "Type review here:",
    height=150,
    placeholder="Example: This product is amazing and works perfectly!"
)

# ==============================
# Prediction
# ==============================
if st.button("Predict Sentiment"):

    if user_review.strip() == "":
        st.warning("⚠️ Please enter a review first.")

    else:
        cleaned_review = clean_text(user_review)

        prediction = model.predict([cleaned_review])[0]

        emoji = {
            "Positive": "😊",
            "Negative": "😡",
            "Neutral": "😐"
        }

        st.success(f"Predicted Sentiment: {emoji[prediction]} {prediction}")

        st.subheader("🧹 Cleaned Review")
        st.write(cleaned_review)

# ==============================
# Sample Reviews
# ==============================
st.subheader("📌 Example Reviews")

examples = [
    "Amazing quality and fast delivery!",
    "Worst product ever. Waste of money.",
    "The product is okay, nothing special."
]

for ex in examples:
    st.write(f"• {ex}")

# ==============================
# Footer
# ==============================
st.markdown("---")
st.markdown("✅ Built using Streamlit, NLP & Machine Learning")
