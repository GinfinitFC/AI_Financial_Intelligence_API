# app/api/routes/stocks.py

from fastapi import APIRouter
from app.services.stock_service import get_stock_history

router = APIRouter(prefix="/stocks")

@router.get("/history")
def history(ticker: str, period: str = "1y"):
    return get_stock_history(ticker, period)