from fastapi import APIRouter, Query
from pydantic import BaseModel

from app.services.llm_service import ask_model
from app.services.analysis_service import analyze_stock, analyze_macro

router = APIRouter(prefix="/analysis")

class QuestionRequest(BaseModel):
    question: str


@router.post("/trending")
def trending_analysis(request: QuestionRequest):

    response = ask_model(request.question)

    return {
        "response": response
    }

@router.get("/stock")
def stock_analysis(
    ticker: str,
    max_articles: int = Query(
        default=10,
        ge=1,
        le=50
    )
):

    return analyze_stock(
        ticker=ticker,
        max_articles=max_articles
    )

@router.get("/macro")
def macro_analysis(
    max_articles: int = Query(
        default=10,
        ge=1,
        le=50
    )
):

    return analyze_macro(
        max_articles=max_articles
    )