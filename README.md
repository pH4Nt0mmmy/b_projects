
# 🧠 Sentiment Analysis Tool with Custom ML Model

Welcome to **Sentiment Analysis Tool**, a machine learning-based text classifier that predicts the **sentiment** (Positive/Negative/Uncertain) of any given English text using a custom-trained model.

---

## 📌 About the Project

This is **my first machine learning project from scratch** where I trained my own model instead of relying on pre-trained Hugging Face or OpenAI models. Until now, I have only experimented with ready-made models. This project represents my first independent attempt at building and deploying a **text classification pipeline** using classical ML algorithms (e.g., Logistic Regression, TF-IDF).

---

## 🚀 Features

- 🔍 Custom model trained on TF-IDF + Scikit-learn
- 🧠 Rule-based enhancements (keyword detection)
- 🌐 English keyword extraction
- 🎨 Terminal output with emoji + colored feedback (via `colorama`)
- 📈 Keeps history of last 5 predictions
- ⚠️ Smart warning for low-confidence predictions
- 💡 Motivation tip generator (based on sentiment)

---

## 🧪 Model Overview

- **Vectorizer**: `TfidfVectorizer` (Scikit-learn)
- **Classifier**: `LogisticRegression` or similar linear classifier
- **Trained with**: Custom labeled text dataset (you can provide your own `.csv`)
- **Storage**: `joblib` for saving the trained model and vectorizer

---

## 🛠️ Installation
* You have to download the dataset but there are some files are missing so please check the links too
  
### 1. Clone this repository

```bash
git clone https://github.com/pH4Nt0mmmy/b_projects.git
cd b_projects
```

### 2. Create virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate (Windows)
```






## ▶️ How to Use

Just run the script:

```bash
python sentiment_tool.py
```

Then enter any text when prompted:

```
Enter your text: I think this product is amazing!
👉 Prediction: Positive 😊 (97.4% confidence) 🚀
🔑 Keywords: product, amazing
💬 Tip: Great! Keep up the good energy!
```

You can also type:
- `q` → quit
- `history` → view last 5 predictions

---



## 📦 Requirements

- Python 3.8+
- `scikit-learn`
- `joblib`
- `colorama`
- `re` (built-in)
- Datasets 

You can install them with:

```bash
pip install -r requirements.txt
```

---

# Links for datasets
 * https://www.kaggle.com/datasets/kazanova/sentiment140
 * https://components.one/datasets/all-the-news-2-news-articles-dataset
 * https://www.kaggle.com/datasets/yogeshchary/financial-news-dataset
---

## 🧠 What I Learned

This project helped me:
- Understand how text is cleaned and vectorized
- Work with real model training and prediction
- Structure a Python project from scratch
- Think beyond accuracy by using rules (e.g., keyword override)
- Manage prediction confidence

---

## Warning
- This is my first project about ML and there are several points to improve it because there are a lot of problems (example: You can get false answers). I would be happy to hear your feedback and suggestions. 

---
##  Signature

```
Project Author: pH4nt0mmmy
```

---

## 🕊️ License

MIT License - Free to use, modify and share.

---

## 🤝 Contributions

Feel free to fork this repository, improve the pipeline or share new datasets!


