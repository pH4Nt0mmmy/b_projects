from scrapping import scrape_google_news
import json
import os
import time

with open("keywords.json", "r", encoding="utf-8") as f:
    keywords_json = json.load(f)

keywords = list(set([kw for category in keywords_json.values() for kw in category]))

all_news = []
failed_urls_all = []
output_path = "news_data.json"

if os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as f:
        all_news = json.load(f)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36'
}

for i, keyword in enumerate(keywords, 1):
    print(f"\n🔍 ({i}/{len(keywords)}) Aranıyor: {keyword}")
    try:
        results, failed_urls = scrape_google_news(keyword, lang="en", max_results=5, headers=headers)
        all_news.extend(results)
        failed_urls_all.extend(failed_urls)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_news, f, ensure_ascii=False, indent=2)
        time.sleep(2)  
    except Exception as e:
        print(f"⚠️ {keyword} aranırken hata oluştu: {e}")
        continue

with open("failed_urls.json", "w", encoding="utf-8") as f:
    json.dump(list(set(failed_urls_all)), f, ensure_ascii=False, indent=2)

print(f"\n✅ Toplam haber sayısı: {len(all_news)} kayıt '{output_path}' dosyasına kaydedildi.")
print(f"[!] Toplam başarısız URL sayısı: {len(set(failed_urls_all))}")