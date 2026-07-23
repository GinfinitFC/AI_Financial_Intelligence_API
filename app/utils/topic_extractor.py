from typing import List
import re


TOPIC_KEYWORDS = {

    # -------------------------
    # Macroeconomics
    # -------------------------

    "inflation": [
        "inflation",
        "cpi",
        "consumer price index",
        "prices rise",
        "cost of living",
    ],

    "federal_reserve": [
        "federal reserve",
        "fed",
        "fomc",
        "jerome powell",
    ],

    "interest_rates": [
        "interest rate",
        "interest rates",
        "rate hike",
        "rate cut",
        "monetary policy",
    ],

    "gdp": [
        "gdp",
        "gross domestic product",
        "economic growth",
    ],

    "unemployment": [
        "unemployment",
        "jobless claims",
        "labor market",
        "employment report",
    ],

    "tariffs": [
        "tariff",
        "trade war",
        "import duties",
    ],


    # -------------------------
    # Technology
    # -------------------------

    "artificial_intelligence": [
        "artificial intelligence",
        "generative ai",
        "machine learning",
        "deep learning",
        "large language model",
        "llm",
        "foundation model",
        "ai model",
        "ai",
    ],

    "semiconductors": [
        "semiconductor",
        "chip",
        "chips",
        "gpu",
        "cpu",
        "foundry",
        "wafer",
        "nvidia",
        "amd",
        "intel",
        "tsmc",
        "broadcom",
    ],

    "cloud_computing": [
        "cloud computing",
        "aws",
        "azure",
        "google cloud",
        "data center",
        "datacenter",
    ],

    "robotics": [
        "robotics",
        "robot",
        "automation",
        "autonomous",
    ],


    # -------------------------
    # Markets
    # -------------------------

    "earnings": [
        "earnings",
        "quarterly results",
        "revenue",
        "profit",
        "eps",
    ],

    "analyst_ratings": [
        "analyst",
        "price target",
        "upgrade",
        "downgrade",
        "rating",
    ],

    "mergers_acquisitions": [
        "acquisition",
        "merger",
        "buyout",
    ],

    "stock_buybacks": [
        "buyback",
        "share repurchase",
    ],


    # -------------------------
    # Energy
    # -------------------------

    "oil_energy": [
        "oil",
        "crude",
        "opec",
        "energy prices",
    ],

    "renewable_energy": [
        "solar",
        "wind energy",
        "renewable",
        "clean energy",
    ],
}

def infer_topics(text: str) -> List[str]:
    """
    Infer topics from article text
    using keyword matching.
    """

    if not text:
        return []

    text = text.lower()

    topics = []

    for topic, keywords in TOPIC_KEYWORDS.items():

        for keyword in keywords:

            # Word boundary avoids false positives
            pattern = rf"\b{re.escape(keyword)}\b"

            if re.search(pattern, text):
                topics.append(topic)
                break

    return topics