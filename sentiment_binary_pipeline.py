import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os

df = pd.read_csv("sentiment_dataset.csv")
df = df[df["label"].isin(["positive", "negative"])].copy()
df["label"] = df["label"].map({"positive": 1, "negative": 0})
df.dropna(subset=["text", "label"], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
)

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_df=0.9,
    min_df=5
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

param_grid = {
    "C": [0.1, 1, 10],
    "penalty": ["l2"],
    "solver": ["lbfgs"]
}
log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
grid = GridSearchCV(log_reg, param_grid, cv=3, scoring="f1", n_jobs=-1)
grid.fit(X_train_vec, y_train)

best_model = grid.best_estimator_

y_pred = best_model.predict(X_test_vec)
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
plt.title("🧠 Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

print("\n🔍 SHAP açıklanabilirlik analizine geçiliyor...")
explainer = shap.Explainer(best_model, X_train_vec[:100])
shap_values = explainer(X_test_vec[:100])
shap.plots.bar(shap_values, max_display=10)

os.makedirs("model", exist_ok=True)
joblib.dump(best_model, "model/sentiment_model.joblib")
joblib.dump(vectorizer, "model/tfidf_vectorizer.joblib")

print("✅ Model ve vektörleştirici 'model/' klasörüne kaydedildi.")