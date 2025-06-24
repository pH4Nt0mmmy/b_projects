import pandas as pd

datasets = []

try:
    df1 = pd.read_csv("sentiment140.csv", encoding="latin1", header=None)
    df1 = df1[[0, 5]]
    df1.columns = ["label", "text"]
    df1["label"] = df1["label"].map({0: "negative", 4: "positive"})
    datasets.append(df1)
except Exception as e:
    print(f"❌ sentiment140.csv okunamadı: {e}")

try:
    df2 = pd.read_csv("Financial_Sentiment.csv")
    df2 = df2[["Content", "Tag"]]
    df2.columns = ["text", "label"]
    datasets.append(df2)
except Exception as e:
    print(f"❌ Financial_Sentiment.csv okunamadı: {e}")

try:
    df3 = pd.read_csv("Financial_Sentiment_Categorized.csv")
    df3 = df3[["Content", "Tag"]]
    df3.columns = ["text", "label"]
    datasets.append(df3)
except Exception as e:
    print(f"❌ Categorized versiyon okunamadı: {e}")


all_data = pd.concat(datasets, ignore_index=True)
all_data = all_data.dropna().drop_duplicates()

all_data["text"] = all_data["text"].str.strip()
all_data = all_data[all_data["text"].str.len() > 10]


all_data.to_csv("sentiment_dataset.csv", index=False)
print(f"✅ Birleştirilmiş veri başarıyla oluşturuldu → sentiment_dataset.csv")
print(all_data["label"].value_counts())