# app/services/macro_service.py

import os

from newsapi import NewsApiClient
from dotenv import load_dotenv
from app.utils.news_normalizer import normalize_newsapi_article
from app.services.sentiment_service import hybrid_sentiment, sentiment_label



loaded = load_dotenv()
newsapi = NewsApiClient(api_key=os.getenv("NEWSAPI_KEY"))

MACRO_TOPICS = [
    "inflation",
    "interest rates",
    "Federal Reserve",
    "GDP",
    "unemployment",
    "tariffs",
    "global markets",
    "oil prices",
    "artificial intelligence",
    "semiconductors"
]

def get_macro_news(vector_service: None, max_articles: int = 20):
    #v1
    query = (
        'inflation OR '
        '"interest rates" OR '
        '"Federal Reserve" OR '
        'GDP OR '
        'unemployment OR '
        'tariffs OR '
        '"global markets" OR '
        '"oil prices" OR '
        '"artificial intelligence" OR '
        'semiconductors'
    )

    response = newsapi.get_everything(
        q=query,
        language="en",
        sort_by="publishedAt",
        page_size=max_articles
    )

    normalized_news = [
        normalize_newsapi_article(article)
        for article in response["articles"]
    ]

    if vector_service:
        vector_service.ingest_documents(normalized_news)

    return normalized_news

def get_macro_sentiment(max_articles: int = 20):
    #Retrieve news articles related to macroeconomic topics
    articles = get_macro_news(
        max_articles=max_articles
    )

    results = []
    scores = []

    #sentiment counts
    positive = 0
    neutral = 0
    negative = 0

    topic_counter = {}

    for article in articles:

        sentiment = hybrid_sentiment(
            article["content"]
        )

        score = sentiment["score"]

        scores.append(score)

        if score >= 0.1:
            positive += 1

        elif score <= -0.1:
            negative += 1

        else:
            neutral += 1

        for topic in article["topics"]:

            topic_counter[topic] = (
                topic_counter.get(topic, 0) + 1
            )

        results.append(
            {
                "title": article["title"],
                "publisher": article["publisher"],
                "summary": article["summary"],
                "topics": article["topics"],
                "method": sentiment["method"],
                "sentiment_score": round(score, 3),
                "sentiment": sentiment_label(score)
            }
        )

    average_sentiment = (
        sum(scores) / len(scores)
        if scores
        else 0
    )

    overall_label = sentiment_label(average_sentiment)

    major_topics = sorted(
        topic_counter,
        key=topic_counter.get,
        reverse=True
    )[:5]

    return {
        "average_sentiment": round(
            average_sentiment,
            3
        ),

        "overall_sentiment": overall_label,

        "positive_articles": positive,
        "neutral_articles": neutral,
        "negative_articles": negative,

        "total_articles": len(articles),

        "major_topics": major_topics,

        "articles": results
    }