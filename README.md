# Financial Intelligence API

An AI-powered financial intelligence backend built with **FastAPI**, combining financial data retrieval, market analysis, NLP-based sentiment analysis, semantic search, and LLM-powered reasoning.

The goal of this project is to build a modular platform capable of transforming raw financial information into structured insights by combining:

- Market data analysis
- Financial news ingestion
- Sentiment analysis
- Vector-based semantic retrieval
- Retrieval-Augmented Generation (RAG)
- LLM reasoning over financial context


# Architecture Overview

The system follows a modular service-oriented architecture:


app
│
├── main.py
│
├── routes
│   ├── analysis.py
│   ├── fx.py
│   ├── health.py
│   ├── news.py
│   ├── sentiment.py
│   └── stocks.py
│
├── services
│   ├── analysis_service.py
│   ├── fx_service.py
│   ├── llm_service.py
│   ├── macro_service.py
│   ├── sentiment_service.py
│   ├── stock_service.py
│   ├── ticker_news_service.py
│   └── vector_service.py
│
└── utils
├── indicators.py
├── news_normalizer.py
└── topic_extractor.py


The application separates:

- API routing
- Business logic
- Data processing
- AI inference
- Vector retrieval
- Utility functions


# Features

## Market Data Analysis

Retrieve and analyze financial market information including:

- Stock prices
- Historical market data
- Moving averages
- Trend indicators
- Volatility analysis
- Golden cross / death cross detection

Supported through financial data providers such as Yahoo Finance.


# Financial News Intelligence

The API retrieves and normalizes financial news from multiple sources.

Each article is transformed into a common internal format:

{
    "id": "article-id",
    "title": "News title",
    "summary": "News summary",
    "content": "Article content",
    "publisher": "Publisher",
    "published": "2026-07-23",
    "asset": "NVDA",
    "topics": [
        "artificial_intelligence",
        "semiconductors"
    ]
}

The normalization layer provides:

* Duplicate detection
* Consistent schema across providers
* Metadata extraction
* Topic classification


# Topic Extraction

News articles are automatically classified using a lightweight keyword-based topic extraction system.

Current supported topics include:

* Artificial Intelligence
* Semiconductors
* Cloud Computing
* Earnings
* Federal Reserve
* Inflation
* Interest Rates
* Renewable Energy
* Oil & Energy
* Analyst Ratings
* Tariffs

The topic system is designed to be extensible and can later be upgraded with:

* Transformer classifiers
* Embedding-based classification
* LLM-assisted categorization


# Sentiment Analysis

The API provides sentiment analysis over financial text.

Current pipeline:

News Article
      |
      v
Text Processing
      |
      v
Sentiment Model
      |
      v
Financial Sentiment Score


The system supports:

* Individual article sentiment
* Aggregated ticker sentiment
* Macro market sentiment


# Vector Database and Semantic Search

The project implements a custom vector retrieval system using:

* Sentence Transformers
* FAISS
* Persistent local vector storage

Embedding model:

all-MiniLM-L6-v2


The vector service provides:

* Document embeddings
* Persistent FAISS indexes
* Metadata storage
* Duplicate prevention
* Similarity search
* Filter-aware retrieval


# Metadata-Based Retrieval

Documents maintain structured metadata indexes to enable advanced filtering.

Supported filters:

* Asset / ticker
* Category
* Publisher
* Source
* Topics
* Publication date range

Example:
SearchFilters(
    asset="NVDA",
    topics=[
        "semiconductors"
    ],
    start_date="2026-07-01",
    end_date="2026-07-23"
)

The retrieval pipeline combines:

Metadata Filtering
        |
        v
FAISS Similarity Search
        |
        v
Ranking
        |
        v
Context Generation

# Retrieval-Augmented Generation (RAG)

The API is evolving toward an RAG-based financial reasoning system.

Current pipeline:

User Query
    |
    v
Semantic Search
    |
    v
Relevant Financial Documents
    |
    v
Context Builder
    |
    v
LLM Reasoning
    |
    v
Financial Insight

The LLM layer is intentionally separated from retrieval, allowing future combination of multiple contexts:

* Asset-specific news
* Macro-economic news
* Technical indicators
* Market sentiment
* Historical events

---

# Vector Service Design

The vector service manages:

## Document Storage

Each document stores:

* Original metadata
* Article content
* Unique identifier
* Asset information

## FAISS Index

The system maintains:

* Vector embeddings
* FAISS-to-document mappings
* Document-to-vector mappings

## Metadata Index

Fast filtering is supported through inverted indexes:

asset
category
publisher
source
topics
published

# Vector Database

The FAISS vector store is generated locally and is not included in the repository.

To rebuild the vector database:

1. Start the API
2. Retrieve financial news
3. Ingest documents through the vector service

The vector index will be automatically generated under:
data/vector_store/

# Tech Stack

## Backend

* Python
* FastAPI
* Pydantic
* Uvicorn

## Data Processing

* Pandas
* NumPy
* Scikit-learn

## Financial Data

* Yahoo Finance API
* News APIs

## NLP / AI

* Sentence Transformers
* FAISS
* LLM APIs

## Development

* Git
* Virtual environments
* Logging
* Modular service architecture

---

# Running the Project

Clone the repository:
git clone https://github.com/GinfinitFC/ai-inference-api.git

Create a virtual environment:
python -m venv venv

Activate environment:

Mac/Linux:

source venv/bin/activate

Install dependencies:
pip install -r requirements.txt

Run API:
uvicorn app.main:app --reload

# Example API Capabilities

## Health Check

GET /health

## Stock Analysis

GET /stocks/{ticker}

Returns:

* Current price
* Indicators
* Historical information

## Financial News Retrieval

GET /news/{ticker}

Returns normalized financial news.

## Semantic Search

Retrieve relevant financial documents based on natural language queries.

Example:

"What is driving Nvidia growth?"

Returns:

* Relevant documents
* Similarity scores
* Metadata
* Ranked results

# Future Roadmap

## Completed

* [x] FastAPI backend
* [x] Modular service architecture
* [x] Stock analysis service
* [x] Financial news ingestion
* [x] Sentiment analysis
* [x] LLM analysis service
* [x] Sentence Transformer embeddings
* [x] FAISS vector database
* [x] Metadata filtering
* [x] Date range filtering
* [x] RAG context generation

## Planned

### Advanced RAG Pipeline

* [ ] Multi-context retrieval
* [ ] Better document ranking
* [ ] Query rewriting
* [ ] Source weighting

### Financial Intelligence

* [ ] Earnings report ingestion
* [ ] SEC filing analysis
* [ ] Company fundamentals
* [ ] Sector analysis
* [ ] Market regime detection

### AI Improvements

* [ ] Transformer-based topic classification
* [ ] Financial-specific embedding models
* [ ] Agent-based financial analysis workflows