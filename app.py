from flask import Flask, request, render_template, jsonify
from transformers import pipeline
from datasets import load_dataset
from tweet_sentiment import get_sentiment_from_tweets
from flipkart_sentiment import get_sentiment_from_flipkart
app = Flask(__name__)
sentiment_model = pipeline("sentiment-analysis")
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        reviews = request.form.get("reviews")
        review_list = [r.strip() for r in reviews.split(",") if r.strip()]
        analysis = sentiment_model(review_list)

        positive = sum(1 for r in analysis if r["label"] == "POSITIVE")
        negative = sum(1 for r in analysis if r["label"] == "NEGATIVE")
        total = len(analysis)
        percentage = round((positive / total) * 100, 2)

        summary = {
            "total": total,
            "positive": positive,
            "negative": negative,
            "percentage": percentage,
            "overall": "POSITIVE" if percentage >= 60 else "NEGATIVE",
            "details": list(zip(review_list, analysis))
        }
        result = summary

    return render_template("index.html", result=result)

@app.route("/get_data", methods=["GET"])
def get_reviews():
    review_text = request.args.get("text")
    review_list = [r.strip() for r in review_text.split(",") if r.strip()]
    analysis = sentiment_model(review_list)
    result = {
        "total": len(analysis),
        "positive": sum(1 for r in analysis if r["label"] == "POSITIVE"),
        "negative": sum(1 for r in analysis if r["label"] == "NEGATIVE"),
        "percentage": round(
            sum(1 for r in analysis if r["label"] == "POSITIVE") / len(analysis) * 100, 2
        ),
        "overall": "POSITIVE" if sum(1 for r in analysis if r["label"] == "POSITIVE") / len(analysis) >= 0.6 else "NEGATIVE",
        "details": list(zip(review_list, analysis))
    }
    return jsonify(result)
@app.route("/get_data_html", methods=["GET", "POST"])
def get_data_html():
    dataset = load_dataset("amazon_polarity", split="train[:10]")
    reviews = [sample["content"] for sample in dataset]
    analysis = sentiment_model(reviews)
    positive = sum(1 for r in analysis if r["label"] == "POSITIVE")
    negative = sum(1 for r in analysis if r["label"] == "NEGATIVE")
    total = len(analysis)
    percentage = round((positive / total) * 100, 2)

    summary = {
        "total": total,
        "positive": positive,
        "negative": negative,
        "percentage": percentage,
        "overall": "POSITIVE" if percentage >= 60 else "NEGATIVE",
        "details": list(zip(reviews, analysis))
    }
    return render_template("index.html", result=summary)

@app.route("/api", methods=["GET"])
def api():
    product = request.args.get("q", "iphone 15 review")
    limit = int(request.args.get("limit", 20))
    result = get_sentiment_from_tweets(product, limit)
    return jsonify(result)

@app.route("/tweet_sentiment", methods=["GET", "POST"])
def tweet_sentiment_page():
    result = None
    if request.method == "POST":
        product = request.form.get("product")
        print("🧪 User entered product:", product)  
        if product:
            result = get_sentiment_from_tweets(product, limit=20)
    return render_template("index.html", result=result)
@app.route("/flipkart_sentiment", methods=["GET", "POST"])
def flipkart_sentiment_page():
    result = None
    if request.method == "POST":
        product = request.form.get("product")
        if product:
            # Correct function call
            result = get_sentiment_from_flipkart(product, limit=100)  # You can adjust limit here
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
