from fastapi import FastAPI
from app.routes import health, fx, stocks, sentiment, analysis, news

app = FastAPI(title="AI Financial Intelligence API")

app.include_router(health.router)
app.include_router(fx.router)
app.include_router(stocks.router)
app.include_router(sentiment.router)
app.include_router(analysis.router)
app.include_router(news.router)


@app.get("/")
def root():
    return {"message": "Financial Intelligence API is running"}