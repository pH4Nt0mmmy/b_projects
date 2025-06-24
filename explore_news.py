import json
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

with open("news_data_cleaned.json", "r", encoding="utf-8") as f:
    data = json.load(f)


df = pd.DataFrame(data)


all_words = []
for text in df["cleaned_text"]:
    tokens = word_tokenize(text)
    all_words.extend(tokens)


word_freq = Counter(all_words).most_common(20)
print("🔠 Most used words:")
for word, freq in word_freq:
    print(f"{word}: {freq}")


text_for_cloud = " ".join(all_words)
wc = WordCloud(width=800, height=400, background_color="white").generate(text_for_cloud)

plt.figure(figsize=(12, 6))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Keywords")
plt.show()


bigrams = list(ngrams(all_words, 2))
bigram_freq = Counter(bigrams).most_common(15)

print("\n🔗 En sık geçen bigramlar:")
for pair, freq in bigram_freq:
    print(f"{pair}: {freq}")

keywords = [word for word in all_words if word in ["china", "usa", "war", "market", "oil", "iran", "crypto"]]
keyword_freq = Counter(keywords)

sns.barplot(x=list(keyword_freq.keys()), y=list(keyword_freq.values()))
plt.title("Keyword Frequency")
plt.ylabel("Frequency")
plt.show()