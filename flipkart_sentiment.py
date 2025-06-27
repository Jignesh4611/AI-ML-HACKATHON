from datasets import load_dataset
from transformers import pipeline

# Load sentiment model once
sentiment_model = pipeline("sentiment-analysis")

# Load Flipkart dataset (train split)
dataset = load_dataset("KayEe/flipkart_sentiment_analysis", split="train")

def get_sentiment_from_flipkart(query="iphone", limit=None):
    """
    Returns sentiment summary for Flipkart reviews matching the given product query.
    No terminal output.
    """
    query = query.lower().replace(" ", "")
    filtered = [row for row in dataset if query in row["input"].lower().replace(" ", "")]

    if not filtered:
        return {
            "product": query,
            "total": 0,
            "positive": 0,
            "negative": 0,
            "percentage": 0.0,
            "overall": "UNKNOWN",
            "details": []
        }

    if limit:
        filtered = filtered[:limit]

    texts = [r["input"] for r in filtered]
    actuals = [r["output"].upper() for r in filtered]

    predictions = sentiment_model(texts)

    positive = sum(1 for p in predictions if p["label"].upper() == "POSITIVE")
    negative = sum(1 for p in predictions if p["label"].upper() == "NEGATIVE")
    percentage = round((positive / len(predictions)) * 100, 2)

    details = [
        {
            "text": text,
            "actual": actual,
            "predicted": pred["label"],
            "confidence": round(pred["score"] * 100, 2)
        }
        for text, actual, pred in zip(texts, actuals, predictions)
    ]

    return {
        "product": query,
        "total": len(texts),
        "positive": positive,
        "negative": negative,
        "percentage": percentage,
        "overall": "POSITIVE" if percentage >= 60 else "NEGATIVE",
        "details": details
    }
