from app.services.stock_service import get_stock_signal
from app.services.ticker_news_service import get_news_sentiment
from app.services.llm_service import analyze_stock_context, analyze_macro_context
from app.services.macro_service import get_macro_sentiment

def build_llm_context(technical_analysis, news_sentiment):
    
    return {
        "current_price": technical_analysis["current_price"],
        "trend": technical_analysis["trend"],
        "golden_cross": technical_analysis["golden_cross"],
        "death_cross": technical_analysis["death_cross"],
        "average_sentiment": news_sentiment["average_sentiment"],
        "overall_sentiment": news_sentiment["overall_sentiment"]
    }

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


def build_macro_context(macro_sentiment: dict):
    context = (
        f"Market Sentiment Analysis:\n"
        f"Overall Sentiment: {macro_sentiment['overall_sentiment']}\n"
        f"Average Sentiment Score: {macro_sentiment['average_sentiment']}\n"
        f"Positive Articles: {macro_sentiment['positive_articles']}\n"
        f"Neutral Articles: {macro_sentiment['neutral_articles']}\n"
        f"Negative Articles: {macro_sentiment['negative_articles']}\n"
        f"Total Articles Analyzed: {macro_sentiment['total_articles']}\n"
        f"Major Topics: {', '.join(macro_sentiment['major_topics'])}\n\n"
    )

    return context

def analyze_macro(max_articles: int = 10):
    
    macro_sentiment = get_macro_sentiment(
        max_articles=max_articles
    )

    context = build_macro_context(macro_sentiment)

    llm_analysis = analyze_macro_context(context)

    return {
        "macro_sentiment": macro_sentiment,
        "llm_analysis": llm_analysis
    }