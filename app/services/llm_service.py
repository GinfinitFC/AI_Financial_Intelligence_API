from openai import OpenAI
import os
import json

from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ask_model(question: str):

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a financial analysis assistant. "
                    "Provide concise bullet point summaries."
                )
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    return response.choices[0].message.content

def analyze_financial_sentiment(text: str):

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": """
                You are a financial sentiment classifier.

                Return ONLY one of:
                positive
                neutral
                negative
                """
            },
            {
                "role": "user",
                "content": text
            }
        ]
    )

    return response.choices[0].message.content.strip().lower()


def analyze_stock_context(context: dict):

    prompt = f"""
        You are a financial analyst.

        Analyze the provided stock information.

        Provide:
        1. A short technical analysis.
        2. A short sentiment analysis.
        3. An overall conclusion.

        Keep the response under 150 words.

        Data:
        {json.dumps(context, indent=2)}
        """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a professional financial analyst."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content