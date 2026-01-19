
# sentiment-based-recommendation-system

## Description
This Python project implements recommendation systems, machine learning models, and NLP processing, and exposes a Flask API. It uses `uv` for virtual environment and package management.

---

## Features

- User-based and item-based recommendation systems
- Machine learning models: Logistic Regression, Random Forest, Naive Bayes
- NLP preprocessing with NLTK, SpaCy, TextBlob
- Model saving and loading using pickle
- Flask API for serving predictions

---

## Requirements

- Python 3.10+
- `uv` (Universal Python Environment)
- Packages listed in `requirements.txt`

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/username/project_name.git
cd project_name


## 2. Create and activate virtual environment using
- uv venv .venv        # Create virtual environment in .venv folder
- source .venv/bin/activate    # macOS/Linux

## Install dependencies
uv pip install -r requirements.txt



# Sentiment-Based Product Recommendation System

---

## Introduction

The e-commerce industry is rapidly growing and transforming the way people shop for products such as books, electronics, cosmetics, medicines, and more. Platforms like **Amazon**, **Flipkart**, **Myntra**, and **Snapdeal** leverage intelligent systems to provide personalized customer experiences and set high benchmarks in online retail.

In this project, we assume the role of a **Senior Machine Learning Engineer** at a growing e-commerce platform. With an expanding product catalog and diverse customer base, the platform aims to improve user engagement and conversion rates by providing **sentiment-aware, personalized product recommendations**.

To remain competitive, the platform must leverage customer review data effectively to understand user preferences and deliver relevant product suggestions. This notebook demonstrates a step-by-step process for building a **Sentiment-Based Product Recommendation System** that combines **textual sentiment analysis** with **collaborative filtering techniques**.

---

## Objectives

The main objectives of this project are:

1. Analyze product review data to understand user sentiments and preferences.
2. Build sentiment classification models using machine learning techniques.
3. Develop a collaborative filtering-based product recommendation system.
4. Integrate sentiment analysis into the recommendation pipeline to refine recommendations.
5. Deploy the system with a user interface for real-time interaction.

---

## Project Workflow

The project is divided into the following stages:

---

### 1. Data Sourcing and Sentiment Analysis

- **Goal**: Preprocess user reviews and train models to classify sentiment.
- **Dataset**: 30,000 reviews across 200+ products from 20,000+ users.
- **Steps**:
  - Exploratory Data Analysis (EDA)
  - Data cleaning and handling missing values
  - Text preprocessing (tokenization, stopword removal, etc.)
  - Feature extraction using:
    - Bag of Words (BoW)
    - TF-IDF vectorization
    - Word embeddings
  - Train and evaluate at least 3 models from the following:
    - Logistic Regression
    - Random Forest
    - XGBoost
    - Naive Bayes
  - Handle class imbalance if required
  - Perform hyperparameter tuning and select the best-performing model

---

### 2. Building the Recommendation System

- **Goal**: Suggest relevant products to users based on past preferences.
- **Techniques**:
  - User-Based Collaborative Filtering
  - Item-Based Collaborative Filtering
- **Steps**:
  - Analyze both approaches and select the best-fit technique
  - Build the recommendation system using user-product rating data
  - For a given user (`reviews_username`), recommend the **top 20 products** likely to be purchased

---

### 3. Sentiment-Enhanced Recommendations

- **Goal**: Refine recommendations based on user sentiment.
- **Steps**:
  - Use the sentiment classification model from Task 1
  - Analyze the sentiment of reviews for the top 20 recommended products
  - Filter and finalize the **top 5 products** with the most positive sentiment scores

---

### 4. Deployment (Optional)

- **Goal**: Deploy the system in a production environment.
- **Suggestions**:
  - Build a UI using **Flask** or **Streamlit**
  - Host on platforms like **Heroku**, **Render**, or **AWS EC2**
  - Allow user input (username or review text) and display personalized recommendations

---

## Deliverables

- Cleaned dataset and EDA report
- Sentiment classification model with performance metrics
- Collaborative filtering-based recommendation system
- Integrated sentiment-aware recommendation pipeline
- (Optional) Web UI for demonstration

---

Let’s start by loading and exploring the dataset!
