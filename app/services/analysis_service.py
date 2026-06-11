from app.services.stock_service import get_stock_signal
from app.services.news_service import get_news_sentiment
from app.services.llm_service import analyze_stock_context


def analyze_stock(ticker: str, max_articles: int = 10):

    #Build context
    technicals = get_stock_signal(
        ticker=ticker
    )

    sentiment = get_news_sentiment(
        ticker=ticker,
        max_articles=max_articles
    )
    context = build_llm_context(technicals, sentiment)

    llm_analysis = analyze_stock_context(context)

    return {
        "ticker": ticker,
        "technical_analysis": technicals,
        "news_sentiment": sentiment,
        "llm_analysis": llm_analysis
    }


def build_llm_context(technical_analysis, news_sentiment):
    
    return {
        "current_price": technical_analysis["current_price"],
        "trend": technical_analysis["trend"],
        "golden_cross": technical_analysis["golden_cross"],
        "death_cross": technical_analysis["death_cross"],
        "average_sentiment": news_sentiment["average_sentiment"],
        "overall_sentiment": news_sentiment["overall_sentiment"]
    }