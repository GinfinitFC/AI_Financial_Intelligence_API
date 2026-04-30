import requests

BASE_URL = "https://open.er-api.com/v6/latest"

def get_fx_rate(from_currency: str, to_currency: str):
    url = f"{BASE_URL}/{from_currency}"
    data = requests.get(url).json()

    if data.get("result") != "success":
        return {"error": "FX API failed", "details": data}

    rate = data["rates"].get(to_currency)

    return {
        "from": from_currency,
        "to": to_currency,
        "rate": rate
    }


def convert_currency(from_currency: str, to_currency: str, amount: float):

    if amount <= 0:
        return {"error": "Amount must be greater than 0"}

    url = f"{BASE_URL}/{from_currency}"
    data = requests.get(url).json()

    if data.get("result") != "success":
        return {"error": "FX API failed", "details": data}

    rate = data["rates"].get(to_currency)

    if rate is None:
        return {"error": f"Conversion rate not found for {to_currency}"}

    converted = amount * rate

    return {
        "from": from_currency,
        "to": to_currency,
        "amount": amount,
        "rate": rate,
        "converted_amount": round(converted, 2)
    }