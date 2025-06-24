import json
import re
import string
from tqdm import tqdm

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  
    text = re.sub(r'\S+@\S+', '', text) 
    text = re.sub(r'\d+', '', text)  
    text = text.translate(str.maketrans("", "", string.punctuation))  
    text = text.strip()
    return text

basic_stopwords = set("""
a about above after again against all am an and any are as at be because been before being below between both but by can did do does doing down during each few for from further had has have having he her here hers herself him himself his how i if in into is it its itself just me more most my myself no nor not of off on once only or other our ours ourselves out over own same she should so some such than that the their theirs them themselves then there these they this those through to too under until up very was we were what when where which while who whom why will with you your yours yourself yourselves
""".split())


def preprocess(text):
    text = clean_text(text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in basic_stopwords and len(t) > 2]
    return " ".join(tokens)


with open("news_data.json", "r", encoding="utf-8") as f:
    news_data = json.load(f)


for item in tqdm(news_data, desc="Preprocessing"):
    item["cleaned_text"] = preprocess(item.get("text", ""))
    item["cleaned_title"] = preprocess(item.get("title", ""))

with open("news_data_cleaned.json", "w", encoding="utf-8") as f:
    json.dump(news_data, f, ensure_ascii=False, indent=2)

print("\n✅ Cleaned data saved on 'news_data_cleaned.json' file.")