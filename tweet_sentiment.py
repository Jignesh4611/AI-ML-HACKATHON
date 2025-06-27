import subprocess
import json
from transformers import pipeline
sentiment_model = pipeline("sentiment-analysis")
def get_sentiment_from_tweets(query, limit=20):
    try:
        command = f"snscrape --jsonl --max-results {limit} twitter-search \"{query}\""  
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)       
        tweets = [
            json.loads(line)["content"]
            for line in result.stdout.strip().split('\n')
            if line.strip()
        ]
        if not tweets:
            return {
                "product": query,
                "total": 0,
                "positive": 0,
                "negative": 0,
                "percentage": 0.0,
                "overall": "NEGATIVE",
                "details": []
            }
        sentiment = sentiment_model(tweets)

        positive = sum(1 for s in sentiment if s["label"] == "POSITIVE")
        negative = sum(1 for s in sentiment if s["label"] == "NEGATIVE")
        percentage = round((positive / len(tweets)) * 100, 2)
        return {
            "product": query,
            "total": len(tweets),
            "positive": positive,
            "negative": negative,
            "percentage": percentage,
            "overall": "POSITIVE" if percentage >= 60 else "NEGATIVE",
            "details": list(zip(tweets, sentiment))
        }
    except subprocess.CalledProcessError as e:
        print("❌ Error running snscrape:", e)
        return {
            "product": query,
            "total": 0,
            "positive": 0,
            "negative": 0,
            "percentage": 0.0,
            "overall": "ERROR",
            "details": []
        }
