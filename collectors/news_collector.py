import os
import json
import time
import logging
import pandas as pd
from tqdm import tqdm
from newspaper import Article
from collectors.util import create_dir, Config

logging.getLogger('newspaper').setLevel(logging.WARNING)


def crawl_article(url, timeout=10):
    if not isinstance(url, str) or url.strip() == "":
        return None

    try:
        # newspaper handles http/https; some sites refuse automated clients
        article = Article(url)
        article.download()
        article.parse()
        return {
            "url": url,
            "title": article.title,
            "text": article.text,
            "authors": article.authors,
            "publish_date": str(article.publish_date),
            "top_image": article.top_image,
            "images": list(article.images)
        }
    except Exception as e:
        logging.warning(f"Failed to fetch {url}: {e}")
        return None


def collect_news_articles(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    create_dir(output_dir)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        news_id = row.get("id")
        url = row.get("news_url")
        save_dir = os.path.join(output_dir, str(news_id))
        create_dir(save_dir)

        content = crawl_article(url)
        if content:
            with open(os.path.join(save_dir, "news_content.json"), "w", encoding="utf-8") as f:
                json.dump(content, f, ensure_ascii=False, indent=2)
        time.sleep(1)  # be polite


def collect_for_choice(choice, config: Config):
    source = choice["news_source"]
    label = choice["label"]
    csv_path = os.path.join(config.dataset_dir, f"{source}_{label}.csv")
    out_dir = os.path.join(config.dump_location, source, label)
    collect_news_articles(csv_path, out_dir)


if __name__ == "__main__":
    # quick local test
    cfg = Config("./dataset", "./output", num_process=1)
    collect_for_choice({"news_source": "gossipcop", "label": "fake"}, cfg)
