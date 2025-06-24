import pandas as pd

files = [
    "sentiment140.csv",
    "Financial_Sentiment.csv",
    "all-the-news-2-1.csv",
    "Financial.csv",
    "Financial_Sentiment_Categorized.csv",
    "financial_news.csv",
    "bitcoin_2017_to_2023.csv"
]

for file in files:
    print(f"\n📄 {file}")
    try:
        df = pd.read_csv(file, encoding="utf-8", nrows=5)
        print(df.columns.tolist())
    except Exception as e:
        print(f"❌ Error reading {file}: {e}")