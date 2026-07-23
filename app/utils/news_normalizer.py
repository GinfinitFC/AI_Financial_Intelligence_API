# app/utils/news_normalizer.py

from typing import Optional
import re
from datetime import datetime
from app.utils.topic_extractor import infer_topics

# ---------------------------------
# Content extraction settings
# ---------------------------------
MAX_CONTENT_CHARS = 2500
MAX_PARAGRAPHS = 10

def normalize_article(
    article: dict,
    category: str,
    asset: Optional[str] = None,
):
    """
    Normalize Yahoo Finance news articles into the
    application's internal document format.
    """

    content = article.get("content", {})

    title = content.get("title") or ""
    summary = content.get("summary") or ""

    # ---------------------------------
    # Extract article body
    # ---------------------------------
    body = []
    current_length = 0

    body_data = content.get("body", {})
    paragraphs = body_data.get("content", [])

    for item in paragraphs:

        if len(body) >= MAX_PARAGRAPHS:
            break

        if item.get("type") != "text":
            continue

        text = item.get("content", "").strip()

        if not text:
            continue

        if current_length + len(text) > MAX_CONTENT_CHARS:
            break

        body.append(text)
        current_length += len(text)

    article_content = "\n\n".join(body)

    # Clean excessive blank lines
    article_content = re.sub(
        r"\n{3,}",
        "\n\n",
        article_content,
    ).strip()

    # Fallback when Yahoo doesn't expose the body
    if not article_content:
        article_content = summary

    #---------------------------------
    # Extract date and time
    #---------------------------------
    published = None
    if content.get("pubDate"):
        published = datetime.strptime(
            content["pubDate"],
            "%Y-%m-%dT%H:%M:%SZ"
        ).date().isoformat()

    return {
        "id": article.get("id"),
        "title": title,
        "summary": summary,
        "content": article_content,
        "publisher": content.get("provider", {}).get("displayName"),
        "link": content.get("canonicalUrl", {}).get("url"),
        "published": published,

        # Metadata
        "category": category,
        "asset": asset,
        "topics": infer_topics(f"{title}\n\n{summary}\n\n{article_content}"),
        "source": "yfinance",
    }

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