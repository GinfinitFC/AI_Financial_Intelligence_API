# app/utils/news_normalizer.py

from typing import Optional


def normalize_article(article: dict, category: str, asset: Optional[str] = None):

    """
    Normalize news articles from different providers
    into a common internal format.
    """

    content = article.get("content", {})
    title = content.get("title") or ""
    summary = content.get("summary") or ""

    return {
        "id": article.get("id"),
        "title": title,
        "summary": summary,
        "content": f"{title}\n\n{summary}".strip(),
        "publisher": content.get("provider", {}).get("displayName"),
        "link": content.get("canonicalUrl", {}).get("url"),
        "published": content.get("pubDate"),

        # Metadata
        "category": category,
        "asset": asset,
        "topics": [],
        "source": "yfinance"
    }