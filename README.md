# 📊 AI Financial Intelligence API

🚧 Work in progress — building toward an AI-powered financial analysis system

A modular **FastAPI backend** for financial data analysis, combining:

* 💱 Currency exchange rates and conversion
* 📈 Stock market historical data
* 🧠 Sentiment analysis
* ⚙️ Technical indicators (SMA, Golden/Death Cross)

This project is designed as a **scalable foundation** for building financial intelligence systems and future AI-driven analysis.

---

## 🚀 Features

* **FX Rates**

  * Get real-time exchange rates
  * Convert between currencies

* **Stock Data**

  * Retrieve historical data using `yfinance`
  * Compute technical indicators

* **Technical Indicators**

  * Simple Moving Averages (SMA 50 / 200)
  * Golden Cross & Death Cross detection

* **Sentiment Analysis**

  * Text-based sentiment scoring (baseline)

---

## 🏗️ Project Structure

```
app/
├── main.py
├── routes/
│   ├── health.py
│   ├── fx.py
│   ├── stocks.py
│   └── sentiment.py
├── services/
│   ├── fx_service.py
│   ├── stock_service.py
│   └── sentiment_service.py
├── utils/
│   └── indicators.py
```

---

## ⚙️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/ai-finance-api.git
cd ai-finance-api

python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

---

## ▶️ Run the API

```bash
uvicorn app.main:app --reload
```

Open docs:
👉 http://127.0.0.1:8000/docs

---

## 🔌 API Endpoints

### Health

```
GET /health
```

### FX Rates

```
GET /fx?from_currency=USD&to_currency=MXN
```

### FX Conversion

```
GET /fx/convert?from_currency=USD&to_currency=MXN&amount=100
```

### Stock History

```
GET /stocks/history?ticker=AAPL&period=1y
```

### Stock Indicators

```
GET /stocks/indicators?ticker=AAPL
```

### Sentiment

```
GET /sentiment?text=Apple is doing great
```

---

## 🧠 Roadmap

* [ ] News ingestion + sentiment per ticker
* [ ] Correlation between sentiment and price movements
* [ ] Aggregated analysis endpoint (`/analysis`)
* [ ] Caching and performance improvements
* [ ] Dockerization

---

## 📌 Tech Stack

* FastAPI
* Pandas / NumPy
* yfinance
* TextBlob
* Requests

---

## 🤝 Contributing

This is a personal project, but contributions and suggestions are welcome.

---

## 📄 License

MIT License

```
```
