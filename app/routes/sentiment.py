# app/api/routes/sentiment.py

from fastapi import APIRouter
from app.services.sentiment_service import analyze_sentiment

router = APIRouter(prefix="/sentiment")

@router.get("/")
def sentiment(text: str):
    return analyze_sentiment(text)