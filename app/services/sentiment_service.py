from textblob import TextBlob

from app.services.llm_service import (
    analyze_financial_sentiment
)

UNCERTAINTY_THRESHOLD = 0.1


def analyze_sentiment(text: str):

    analysis = TextBlob(text)

    return {
        "polarity": analysis.sentiment.polarity,
        "subjectivity": analysis.sentiment.subjectivity
    }


def sentiment_label(score: float):

    if score >= 0.1:
        return "positive"

    if score <= -0.1:
        return "negative"

    return "neutral"


def hybrid_sentiment(text: str):

    sentiment = analyze_sentiment(text)

    score = sentiment["polarity"]

    if abs(score) >= UNCERTAINTY_THRESHOLD:

        return {
            "method": "textblob",
            "score": score,
            "label": sentiment_label(score)
        }

    llm_label = analyze_financial_sentiment(text)

    return {
        "method": "llm",
        "score": score,
        "label": llm_label
    }