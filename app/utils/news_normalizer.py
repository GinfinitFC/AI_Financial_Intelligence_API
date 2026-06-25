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

def infer_topics(text: str):

    text = text.lower()

    topics = []

    if "inflation" in text:
        topics.append("inflation")

    if "federal reserve" in text:
        topics.append("federal_reserve")

    if "interest rate" in text:
        topics.append("interest_rates")

    if "gdp" in text:
        topics.append("gdp")

    if "unemployment" in text:
        topics.append("unemployment")
    
    if "artificial intelligence" in text or "ai" in text:
        topics.append("artificial_intelligence")

    if "semiconductors" in text:
        topics.append("semiconductors")
        
    return topics

def normalize_newsapi_article(article: dict):

    title = article.get("title") or ""
    summary = article.get("description") or ""
    topics = infer_topics(f"{title} {summary}")

    return {
        "id": article.get("url"),

        "title": title,
        "summary": summary,

        "content": f"{title}\n\n{summary}".strip(),

        "publisher": article.get(
            "source",
            {}
        ).get("name"),

        "link": article.get("url"),

        "published": article.get(
            "publishedAt"
        ),

        "category": "macro",
        "asset": None,

        "topics": topics,

        "source": "newsapi"
    }