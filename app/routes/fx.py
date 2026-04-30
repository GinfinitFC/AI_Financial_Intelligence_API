from fastapi import APIRouter
from app.services.fx_service import get_fx_rate, convert_currency

router = APIRouter(prefix="/fx")

@router.get("/")
def fx(from_currency: str, to_currency: str):
    return get_fx_rate(from_currency.upper(), to_currency.upper())


@router.get("/convert")
def convert(from_currency: str, to_currency: str, amount: float):
    return convert_currency(
        from_currency.upper(),
        to_currency.upper(),
        amount
    )