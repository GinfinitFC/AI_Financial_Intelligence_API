import yfinance as yf
from app.services.sentiment_service import hybrid_sentiment, sentiment_label
from app.utils.news_normalizer import normalize_article


def get_news(ticker: str, vector_service: None, max_articles: int = 10):

    stock = yf.Ticker(ticker)

    news = stock.news

    normalized_news = [
        normalize_article(article, category="ticker", asset=ticker)
        for article in news[:max_articles]
    ]

    if vector_service:
        vector_service.ingest_documents(normalized_news)

    return normalized_news

def get_news_sentiment(ticker: str, max_articles: int = 10):

    articles = get_news(
        ticker=ticker,
        max_articles=max_articles
    )

    results = []

    scores = []

    positive = 0
    neutral = 0
    negative = 0

    for article in articles:

        text = f"""
                Title:
                {article['title']}

                Summary:
                {article['summary']}
                """

        sentiment = hybrid_sentiment(text)

        score = sentiment["score"]
        label = sentiment["label"]

        scores.append(score)

        if label == "positive":
            positive += 1
        elif label == "negative":
            negative += 1
        else:
            neutral += 1

        results.append(
            {
                "title": article["title"],
                "summary": article["summary"],
                "method": sentiment["method"],
                "sentiment_score": round(score, 3),
                "sentiment": label
            }
        )

    average_sentiment = (
        sum(scores) / len(scores)
        if scores
        else 0
    )

    overall_label = sentiment_label(
    average_sentiment
    )

    return {
        "ticker": ticker,
        "average_sentiment": round(
            average_sentiment,
            3
        ),
        "overall_sentiment": overall_label,
        "positive_articles": positive,
        "neutral_articles": neutral,
        "negative_articles": negative,
        "total_articles": len(articles),
        "articles": results
    }