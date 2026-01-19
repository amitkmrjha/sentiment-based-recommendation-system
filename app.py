from flask import Flask, request, jsonify, render_template
import pandas as pd
from model import load_models_and_data, check_username, get_recommendations_for_user

app = Flask(__name__)

# Load necessary models and dataset
recommendation_model, tfidf_vectorizer, sentiment_model, data = load_models_and_data()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    username = request.form['username']

    # Check if username exists
    if not check_username(username, data):
        return render_template('index.html', message="Username not found!")

    # Generate recommendations for the user
    top_5_products = get_recommendations_for_user(username, data, tfidf_vectorizer, sentiment_model)

    return render_template('index.html', username=username, recommendations=top_5_products)

if __name__ == "__main__":
    app.run(debug=True)