from newspaper import Article
from googlesearch import search
import time
import requests

import requests
from newspaper import Article
import time

def scrape_google_news(query, lang="en", max_results=10, delay=1, headers=None, timeout=240):
    news_list = []
    urls = []
    if headers is None:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36'
        }

    try:
        for url in search(query, lang=lang):
            urls.append(url)
            if len(urls) >= max_results:
                break
    except Exception as e:
        print(f"🔍 Google araması başarısız: {e}")
        return [], []

    session = requests.Session()
    failed_urls = []

    for url in urls:
        try:
            response = session.get(url, headers=headers, timeout=timeout)
            if response.status_code == 403:
                print(f"[x] 403 Forbidden: {url}")
                failed_urls.append(url)
                continue

            article = Article(url, language=lang)
            article.download(input_html=response.text)
            article.parse()

            news = {
                "title": article.title,
                "text": article.text,
                "url": url,
                "publish_date": str(article.publish_date) if article.publish_date else "Unknown"
            }

            news_list.append(news)
            print(f"[✓] Eklendi: {article.title}")

        except requests.exceptions.Timeout:
            print(f"[x] Zaman aşımı (timeout) oldu: {url}")
            failed_urls.append(url)
        except Exception as error:
            print(f"[x] Hata ({url}): {error}")
            failed_urls.append(url)

        time.sleep(delay)

    return news_list, failed_urls