from app.services.stock_service import get_stock_signal
from app.services.news_service import get_news_sentiment


def analyze_stock(ticker: str, max_articles: int = 10):

    technicals = get_stock_signal(
        ticker=ticker
    )

    sentiment = get_news_sentiment(
        ticker=ticker,
        max_articles=max_articles
    )

    return {
        "ticker": ticker,
        "technical_analysis": technicals,
        "news_sentiment": sentiment
    }