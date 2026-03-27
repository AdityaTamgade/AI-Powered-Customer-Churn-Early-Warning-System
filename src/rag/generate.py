import requests
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found in environment variables")

def generate_response(prediction, retrieved_docs):
    context = "\n".join(retrieved_docs)

    prompt = f"""
    You are a customer churn expert.

    Prediction: {prediction}

    Context:
    {context}

    Explain why the customer may churn and suggest business actions.
    """

    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}",
        json={
            "contents": [{"parts": [{"text": prompt}]}]
        }
    )

    return response.json()["candidates"][0]["content"]["parts"][0]["text"]