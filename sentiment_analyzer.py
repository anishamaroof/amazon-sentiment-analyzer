import streamlit as st
import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# =========================
# APP UI
# =========================
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🛍️")
st.title("🛍️ Amazon Sentiment Analyzer")
st.write("Enter a review and predict sentiment (Positive / Negative / Neutral)")

# =========================
# DATASET
# =========================
positive = [
    "This product is amazing", "Best purchase ever", "Very good quality", "I love it"
]
negative = [
    "Worst product ever", "Very bad quality", "Totally useless", "I hate it"
]
neutral = [
    "It is okay", "Average product", "Not bad not good", "Fine product"
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
# TRAIN MODEL
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"],
    df["sentiment"],
    test_size=0.2,
    random_state=42,
    stratify=df["sentiment"]
)

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

# accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))

st.sidebar.write("📊 Model Accuracy")
st.sidebar.write(f"{accuracy*100:.2f}%")

# =========================
# INPUT
# =========================
user_input = st.text_area("Enter your review")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter text")
    else:
        cleaned = clean_text(user_input)
        prediction = model.predict([cleaned])[0]

        st.success(f"Prediction: {prediction}")

# =========================
# SAMPLE
# =========================
st.write("### Example Reviews")
st.write("- This product is really good")
st.write("- Worst experience ever")
st.write("- It is fine")
