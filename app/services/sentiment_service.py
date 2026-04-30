# app/services/sentiment_service.py

from textblob import TextBlob

def analyze_sentiment(text: str):
    blob = TextBlob(text)

    return {
        "polarity": blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity
    }