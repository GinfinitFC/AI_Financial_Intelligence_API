from openai import OpenAI
import os

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