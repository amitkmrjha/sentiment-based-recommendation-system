import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb

# Load necessary models and dataset
def load_models_and_data():
    with open('pickle_files/recommendation_model.pkl', 'rb') as f:
        recommendation_model = pickle.load(f)

    with open('pickle_files/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)

    with open('pickle_files/xgboost_model.pkl', 'rb') as f:
        sentiment_model = pickle.load(f)

    data = pd.read_csv('data\sample30.csv')

    return recommendation_model, tfidf_vectorizer, sentiment_model, data

# Function to check if a username exists in the dataset
def check_username(username, data):
    return username in data['reviews_username'].unique()


# Function to get recommended products based on user reviews
def get_top_20_recommendations(user_reviewed_products, data):
    # Find similar users based on the products they have reviewed
    similar_users = data[data['id'].isin(user_reviewed_products)]['reviews_username'].unique()
    # Get top 20 products reviewed by similar users
    recommended_products = data[data['reviews_username'].isin(similar_users)]['id'].value_counts().head(20).index.tolist()
    return recommended_products


# Function to calculate sentiment score for a product
def get_sentiment_score(product_reviews, tfidf_vectorizer, sentiment_model):
    if product_reviews.empty:
        return 0

    # Transform reviews into TF-IDF features
    tfidf_features = tfidf_vectorizer.transform(product_reviews)

    # Predict sentiments using the sentiment model
    sentiments = sentiment_model.predict(tfidf_features)

    # Calculate the positive sentiment percentage
    positive_sentiments = sum(sentiments)
    return positive_sentiments / len(sentiments)  # Return as percentage


# Function to generate recommendations based on sentiment analysis
def generate_recommendations(recommended_products, data, tfidf_vectorizer, sentiment_model):
    top_5_products = []

    for product_id in recommended_products:
        product_reviews = data[data['id'] == product_id]['reviews_text']
        product_name = data[data['id'] == product_id]['name'].iloc[0]

        # Get sentiment score for the product
        sentiment_score = get_sentiment_score(product_reviews, tfidf_vectorizer, sentiment_model)

        # Add product name and sentiment score to the list
        top_5_products.append((product_name, sentiment_score))

    # Sort products by sentiment score and return top 5
    top_5_products = sorted(top_5_products, key=lambda x: x[1], reverse=True)[:5]
    return [prod[0] for prod in top_5_products]  # Return only the product names


# Main function to get recommendations for a user
def get_recommendations_for_user(username, data, tfidf_vectorizer, sentiment_model):
    # Filter user-specific data
    user_data = data[data['reviews_username'] == username]

    # Get all products reviewed by the user
    user_reviewed_products = user_data['id'].unique()

    # Get top 20 recommended products based on similar user reviews
    recommended_products = get_top_20_recommendations(user_reviewed_products, data)

    # Generate top 5 products based on sentiment analysis
    top_5_products = generate_recommendations(recommended_products, data, tfidf_vectorizer, sentiment_model)

    return top_5_products