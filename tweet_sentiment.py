import tweepy
from transformers import pipeline

# Your bearer token directly in code (replace with your actual token)
bearer_token = "AAAAAAAAAAAAAAAAAAAAAOgs2wEAAAAAfWplEIRrKAOuF7dEsuwjB0WiSkw%3DLLsMHCB78TfLB4xuJeeH7D0UdixxDoVolovENp5Fz5cYjXds4y"

# Create Twitter client
client = tweepy.Client(bearer_token=bearer_token)

# Load Hugging Face sentiment analysis model
sentiment_model = pipeline("sentiment-analysis")

def get_sentiment_from_tweets(query, limit=20):
    tweets = []

    for tweet in tweepy.Paginator(
        client.search_recent_tweets,
        query=query,
        tweet_fields=["text"],
        max_results=100
    ).flatten(limit=limit):
        tweets.append(tweet.text)

    if not tweets:
        return {
            "product": query,
            "total": 0,
            "positive": 0,
            "negative": 0,
            "percentage": 0.0,
            "overall": "UNKNOWN",
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

# Test run
if __name__ == "__main__":
    result = get_sentiment_from_tweets("Python", limit=10)
    print(result)
