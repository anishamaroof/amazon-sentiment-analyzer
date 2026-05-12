# 🎯 Amazon Review Sentiment Analyzer
### Text Mining — Undergraduate Project

---

## 📌 Project Overview
This project performs **3-class Sentiment Analysis** (Positive / Negative / Neutral)
on Amazon-style product reviews using classic NLP + Machine Learning techniques.

---

## 🧠 Techniques Used
| Area | Method |
|------|--------|
| Text Cleaning | Regex, Lowercasing, Stop-word Removal |
| Tokenization | NLTK word_tokenize |
| Lemmatization | WordNetLemmatizer |
| Feature Extraction | TF-IDF (Unigrams + Bigrams) |
| Models | Logistic Regression, Naive Bayes, Linear SVM |
| Evaluation | Accuracy, Precision, Recall, F1, Confusion Matrix |
| Visualization | Matplotlib, Seaborn |

---

## 📁 Project Structure
```
sentiment_analysis/
├── sentiment_analyzer.py   ← Main script (run this)
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
└── plots/                  ← Auto-generated visualizations
    ├── 01_sentiment_distribution.png
    ├── 02_model_comparison.png
    ├── 03_confusion_matrix.png
    ├── 04_length_distribution.png
    └── 05_top_features.png
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the project
```bash
python sentiment_analyzer.py
```

### 3. What happens
- Dataset is auto-generated (no download needed)
- Text is cleaned and preprocessed
- 3 ML models are trained and compared
- 5 visualizations are saved to `./plots/`
- Live demo shows predictions on sample reviews
- Interactive mode: type your own review and get a prediction!

---

## 📊 Output You Will See
1. **Console output** — accuracy of each model, classification report
2. **5 PNG charts** in `./plots/` folder
3. **Live predictions** on demo reviews
4. **Interactive prompt** — type any review and see the sentiment

---

## 🔑 Key Concepts (for report writing)

### TF-IDF (Term Frequency–Inverse Document Frequency)
Converts text into numerical features. Words that appear often in one review
but rarely across all reviews get higher scores — making them more "informative."

### Logistic Regression
A classification algorithm that learns weights for each word feature
to predict the probability of each sentiment class.

### Naive Bayes
Based on Bayes' theorem. Assumes each word contributes independently.
Very fast and surprisingly effective for text classification.

### Linear SVM
Finds the best hyperplane that separates sentiment classes in feature space.
Usually the strongest performer on text data.

---

## 📈 Expected Results
- Accuracy: **85–95%** depending on model
- Best model: typically **Logistic Regression** or **Linear SVM**

---

## 🎓 Project Contributions
- Text preprocessing pipeline
- Multi-model comparison framework
- Visualization dashboard (5 charts)
- Interactive real-time prediction system

---

*Built for undergraduate Text Mining coursework.*
