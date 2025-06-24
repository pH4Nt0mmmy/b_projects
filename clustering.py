import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

with open("news_data_cleaned.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [item.get("cleaned_text", "") for item in data]

vectorizer = TfidfVectorizer(
    max_df=0.8,
    min_df=5,
    stop_words="english",
    ngram_range=(1, 2)
)
X = vectorizer.fit_transform(texts)

num_clusters = 6
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
kmeans.fit(X)
labels = kmeans.labels_


for i, item in enumerate(data):
    item["cluster"] = int(labels[i])


score = silhouette_score(X, labels)
print(f"\n📈 Silhouette Score: {score:.4f}")

terms = vectorizer.get_feature_names_out()
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

print("\n📚 Cluster Titles:")
for i in range(num_clusters):
    top_terms = [terms[ind] for ind in order_centroids[i, :5]]
    print(f"Cluster {i+1}: {', '.join(top_terms)}")


print("\n📰 Example Titles by Cluster:")
cluster_examples = defaultdict(list)
for i, item in enumerate(data):
    if len(cluster_examples[item["cluster"]]) < 3:
        cluster_examples[item["cluster"]].append(item["title"])

for cluster_id in range(num_clusters):
    print(f"\nCluster {cluster_id + 1}:")
    for title in cluster_examples[cluster_id]:
        print(f" - {title}")

with open("news_data_clustered.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())

plt.figure(figsize=(10, 6))
for i in range(num_clusters):
    plt.scatter(
        X_pca[labels == i, 0],
        X_pca[labels == i, 1],
        label=f"Cluster {i+1}",
        alpha=0.6
    )
plt.title("KMeans Clustering Results (PCA 2D)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()