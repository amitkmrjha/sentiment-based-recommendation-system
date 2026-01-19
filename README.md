# Sentiment-Based Recommendation System

## Overview

This project aims to enhance the product recommendation capabilities of an e-commerce platform by incorporating sentiment analysis of user reviews. By understanding the sentiments expressed in customer feedback, the system delivers more personalized and accurate product suggestions, thereby improving user satisfaction and engagement.

## Table of Contents

- [Background](#background)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Sentiment Analysis](#sentiment-analysis)
  - [Text Vectorization](#text-vectorization)
  - [Handling Class Imbalance](#handling-class-imbalance)
  - [Model Training and Evaluation](#model-training-and-evaluation)
- [Recommendation System](#recommendation-system)
  - [Collaborative Filtering](#collaborative-filtering)
  - [Integration with Sentiment Analysis](#integration-with-sentiment-analysis)
- [Deployment](#deployment)
- [Usage](#usage)
- [Conclusion](#conclusion)

## Background

In the competitive landscape of e-commerce, providing personalized product recommendations is crucial for enhancing the user experience. Traditional recommendation systems often rely on user behavior and purchase history. However, by analyzing the sentiments expressed in user reviews, we can gain deeper insights into customer preferences and dissatisfaction, leading to more refined recommendations.

## Dataset

The dataset used in this project comprises 30,000 reviews spanning over 200 different products, contributed by more than 20,000 users. Each entry in the dataset includes:

- **Review Title**: The headline of the user's review.
- **Review Text**: The detailed content of the user's review.
- **Rating**: The numerical rating provided by the user.
- **User Sentiment**: The sentiment derived from the review (positive or negative).

*Note: The dataset and its attribute descriptions are located in the `data` folder.*

## Data Preprocessing

Effective data preprocessing is vital for accurate sentiment analysis. The following steps were undertaken:

1. **Data Cleaning**: Removal of null values, duplicates, and irrelevant content to ensure data quality.
2. **Text Preprocessing**: Conversion of text to lowercase, removal of punctuation, stop words, and lemmatization to standardize the textual data.
3. **Exploratory Data Analysis (EDA)**: Visualization of data distributions, common word frequencies, and rating patterns to understand underlying trends.

## Sentiment Analysis

To determine the sentiment associated with each user review, the following approach was adopted:

### Text Vectorization

The textual data, comprising a combination of `review_title` and `review_text`, was transformed into numerical representations using the Term Frequency-Inverse Document Frequency (TF-IDF) vectorizer. This technique measures the importance of words relative to the entire dataset, facilitating effective model training.

### Handling Class Imbalance

The dataset exhibited a class imbalance, with a disproportionate number of positive and negative sentiments. To address this, the Synthetic Minority Over-sampling Technique (SMOTE) was employed to balance the classes before model training.

### Model Training and Evaluation

Several machine learning models were trained to classify user sentiments:

- **Logistic Regression**
- **Random Forest**
- **XGBoost**
- **Naive Bayes**

Each model was evaluated using metrics such as Accuracy, Precision, Recall, F1 Score, and Area Under the Curve (AUC). The XGBoost model emerged as the best performer based on these evaluation metrics.

## Recommendation System

Building upon the sentiment analysis, a recommendation system was developed using collaborative filtering techniques.

### Collaborative Filtering

Both user-user and item-item collaborative filtering approaches were explored. The system predicts user preferences based on similarities with other users or items. The Root Mean Square Error (RMSE) metric was utilized to evaluate the performance of these models.

### Integration with Sentiment Analysis

To enhance the recommendation quality, sentiment analysis results were integrated. The top 20 products identified by the recommendation system were further filtered based on user sentiments. The system highlighted the top 5 products with the highest positive sentiments, ensuring recommendations align with user satisfaction.

## Deployment

The project was deployed with a user-friendly interface to facilitate seamless interaction. The deployment stack includes:

- **Backend**: Flask framework to handle API requests and serve the recommendation engine.
- **Frontend**: HTML/CSS templates to render the user interface.
- **Model Serving**: Serialized models using pickle for efficient loading and inference.

*Note: Deployment scripts and configurations are available in the repository.*

## Usage

To utilize the sentiment-based recommendation system:

1. **Clone the Repository**: `git clone https://github.com/ManjitSingh2003/sentiment-based-recommendation-system.git`
2. **Install Dependencies**: `pip install -r requirements.txt`
3. **Run the Application**: `python app.py`
4. **Access the Interface**: Navigate to `http://localhost:5000` in your web browser.

Users can input their preferences and receive personalized product recommendations based on sentiment analysis.

## Conclusion

By integrating sentiment analysis into the recommendation system, this project demonstrates an effective approach to enhancing personalization in e-commerce platforms. Understanding user sentiments allows for more accurate and satisfying product suggestions, ultimately contributing to improved customer experience and loyalty.

*For detailed code and implementation, please refer to the repository files.* 
