from fastapi import APIRouter
from app.services.ticker_news_service import get_news, get_news_sentiment
from app.services.macro_service import get_macro_news, get_macro_sentiment
from fastapi import Query
from app.services.dependencies import vector_service

router = APIRouter(prefix="/news")

@router.get("/stocks")
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
        vector_service=vector_service,
        max_articles=max_articles
    )

@router.get("/stocks/sentiment")
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

@router.get("/macro")
def macro_news(
    max_articles: int = Query(
        default=10,
        ge=1,
        le=100
    )
):
    return get_macro_news(
        vector_service=vector_service,
        max_articles=max_articles
    )

@router.get("/macro/sentiment")
def macro_news_sentiment(
    max_articles: int = Query(
        default=10,
        ge=1,
        le=100
    )
):
    return get_macro_sentiment(
        max_articles=max_articles
    )