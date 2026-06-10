from fastapi import APIRouter
from app.services.news_service import get_news, get_news_sentiment
from fastapi import Query

router = APIRouter(prefix="/news")

@router.get("/")
def news(
    ticker: str,
    max_articles: int = Query(
        default=10,
        ge=1,
        le=100
    )
):
    return get_news(
        ticker=ticker,
        max_articles=max_articles
    )

@router.get("/sentiment")
def news_sentiment(
    ticker: str,
    max_articles: int = Query(
        default=10,
        ge=1,
        le=100
    )
):

    return get_news_sentiment(
        ticker=ticker,
        max_articles=max_articles
    )