#!/usr/bin/env python
# coding: utf-8

# # Sentiment-Based Product Recommendation System
# 
# ---
# 
# ## Introduction
# 
# The e-commerce industry is rapidly growing and transforming the way people shop for products such as books, electronics, cosmetics, medicines, and more. Platforms like **Amazon**, **Flipkart**, **Myntra**, and **Snapdeal** leverage intelligent systems to provide personalized customer experiences and set high benchmarks in online retail.
# 
# In this project, we assume the role of a **Senior Machine Learning Engineer** at a growing e-commerce platform. With an expanding product catalog and diverse customer base, the platform aims to improve user engagement and conversion rates by providing **sentiment-aware, personalized product recommendations**.
# 
# To remain competitive, the platform must leverage customer review data effectively to understand user preferences and deliver relevant product suggestions. This notebook demonstrates a step-by-step process for building a **Sentiment-Based Product Recommendation System** that combines **textual sentiment analysis** with **collaborative filtering techniques**.
# 
# ---
# 
# ## Objectives
# 
# The main objectives of this project are:
# 
# 1. Analyze product review data to understand user sentiments and preferences.
# 2. Build sentiment classification models using machine learning techniques.
# 3. Develop a collaborative filtering-based product recommendation system.
# 4. Integrate sentiment analysis into the recommendation pipeline to refine recommendations.
# 5. Deploy the system with a user interface for real-time interaction.
# 
# ---
# 
# ## Project Workflow
# 
# The project is divided into the following stages:
# 
# ---
# 
# ### 1. Data Sourcing and Sentiment Analysis
# 
# - **Goal**: Preprocess user reviews and train models to classify sentiment.
# - **Dataset**: 30,000 reviews across 200+ products from 20,000+ users.
# - **Steps**:
#   - Exploratory Data Analysis (EDA)
#   - Data cleaning and handling missing values
#   - Text preprocessing (tokenization, stopword removal, etc.)
#   - Feature extraction using:
#     - Bag of Words (BoW)
#     - TF-IDF vectorization
#     - Word embeddings
#   - Train and evaluate at least 3 models from the following:
#     - Logistic Regression
#     - Random Forest
#     - XGBoost
#     - Naive Bayes
#   - Handle class imbalance if required
#   - Perform hyperparameter tuning and select the best-performing model
# 
# ---
# 
# ### 2. Building the Recommendation System
# 
# - **Goal**: Suggest relevant products to users based on past preferences.
# - **Techniques**:
#   - User-Based Collaborative Filtering
#   - Item-Based Collaborative Filtering
# - **Steps**:
#   - Analyze both approaches and select the best-fit technique
#   - Build the recommendation system using user-product rating data
#   - For a given user (`reviews_username`), recommend the **top 20 products** likely to be purchased
# 
# ---
# 
# ### 3. Sentiment-Enhanced Recommendations
# 
# - **Goal**: Refine recommendations based on user sentiment.
# - **Steps**:
#   - Use the sentiment classification model from Task 1
#   - Analyze the sentiment of reviews for the top 20 recommended products
#   - Filter and finalize the **top 5 products** with the most positive sentiment scores
# 
# ---
# 
# ### 4. Deployment 
# 
# - **Goal**: Deploy the system in a cloud environment.
# - **Suggestions**:
#   - Build a UI using **Flask** or **Streamlit**
#   - Host on platforms like **Heroku**, **Render**, or **AWS EC2**
#   - Allow user input (username or review text) and display personalized recommendations
# 
# ---
# 
# ## Deliverables
# 
# - Cleaned dataset and EDA report
# - Sentiment classification model with performance metrics
# - Collaborative filtering-based recommendation system
# - Integrated sentiment-aware recommendation pipeline
# - Web UI for demonstration
# 
# ---
# 
# Let’s start by loading and exploring the dataset!
# 

# In[3]:


import pandas as pd
from dateutil import parser
from tqdm import tqdm

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer ,PorterStemmer
import nltk
import re
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import (
    learning_curve,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split
)
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_halving_search_cv  # Needed for HalvingGridSearchCV
from sklearn.model_selection import HalvingGridSearchCV
from xgboost import XGBClassifier

# Ensure all required resources are present
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# In[4]:


import pickle
import os
from typing import Dict

pickles_dir = "recommendation_app/pickles"


# In[5]:


import pickle
import os

def save_model(obj, name: str, dir_path: str = pickles_dir) -> None:
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, f"{name}.pkl")
    
    try:
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        print(f" Saved: {file_path}")
    except Exception as e:
        print(f" Error saving {name}: {e}")



# In[6]:


def load_model(name: str, dir_path: str = pickles_dir):
    file_path = os.path.join(dir_path, f"{name}.pkl")
    try:
        with open(file_path, "rb") as f:
            obj = pickle.load(f)
        print(f" Loaded: {file_path}")
        return obj
    except Exception as e:
        print(f" Error loading {name}: {e}")
        return None



# In[7]:


# Read the CSV file into a pandas DataFrame
df = pd.read_csv('data/sample30.csv')


# ## Initial Data Checks & Cleaning
# - Quick diagnostics: missing value counts, duplicate rows, outlier detection  
# - Cleaning tasks: fill/drop nulls, unify string patterns, harmonize formats  
# - Why: Clean datasets enable robust downstream analysis and modeling

# In[8]:


# Basic Dataset Overview
def dataset_overview(df):
    """Displays shape, info, and first rows of the dataset."""
    print("Shape:", df.shape)
    display(df.head())
    print(df.info())

# Missing Values Summary
def missing_values_summary(df):
    """Returns missing value count and percentage per column."""
    missing_df = pd.DataFrame({
        "Missing Count": df.isnull().sum(),
        "Missing Percentage (%)": (df.isnull().mean() * 100).round(2)
    })
    return missing_df[missing_df["Missing Count"] > 0].sort_values(
        by="Missing Count", ascending=False
    )

# Duplicate Records
def duplicate_summary(df):
    """Returns number of duplicate rows."""
    return df.duplicated().sum()

# Numerical Summary Statistics
def numerical_summary(df):
    """Returns descriptive statistics for numerical columns."""
    return df.describe().T
    
# Categorical Summary Statistics   
def categorical_summary(df):
    """Returns descriptive statistics for categorical columns."""
    return df.describe(include="object").T
    
#  Unique Value Count    
def unique_value_summary(df):
    """Returns unique value count and data types."""
    return pd.DataFrame({
        "Unique Values": df.nunique(),
        "Data Type": df.dtypes
    }).sort_values("Unique Values", ascending=False)



# In[9]:


dataset_overview(df)


# In[10]:


display(unique_value_summary(df))


# In[11]:


display(missing_values_summary(df))


# In[12]:


print("Duplicates:", duplicate_summary(df))


# In[13]:


display(numerical_summary(df))


# In[14]:


display(unique_value_summary(df))


# In[15]:


# Numerical Distribution Plot
import matplotlib.pyplot as plt
import seaborn as sns

def plot_numerical_distribution(df):
    """Plots histograms for numerical columns."""
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in num_cols:
        plt.figure(figsize=(5, 3))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(col)
        plt.show()
        
# Outlier Detection (Boxplots)
def plot_outliers(df):
    """Plots boxplots for numerical columns."""
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in num_cols:
        plt.figure(figsize=(5, 2))
        sns.boxplot(x=df[col])
        plt.title(col)
        plt.show()


# In[16]:


plot_numerical_distribution(df)


# In[17]:


plot_outliers(df)


# In[18]:


# Checking Distribution of `reviews_rating` column

plt.figure(figsize=(12, 6))
sns.countplot(
    y='reviews_rating',
    data=df,
    hue='reviews_rating',
    palette='Set2',
    legend=False
)
plt.title("Distribution of Reviews Rating by Count")
plt.xlabel("Review Count")
plt.ylabel("Review Rating")
plt.tight_layout()
plt.show()


# The majority of user ratings are skewed toward the higher end, with 5-star ratings being the most frequent. This indicates a potential class imbalance, which could affect model performance—especially since user_sentiment is expected to align closely with these ratings. 

# In[19]:


# Get top 5 brands for each sentiment
top_positive = df[df.user_sentiment == 'Positive'].brand.value_counts(normalize=True, ascending=False).head(5)
top_negative = df[df.user_sentiment == 'Negative'].brand.value_counts(normalize=True, ascending=False).head(5)

# Combine into one DataFrame
top_brands_combined = pd.DataFrame({
    'Positive': top_positive,
    'Negative': top_negative
}).fillna(0)

# Plot
top_brands_combined.plot(kind='barh', figsize=(12,6))
plt.title("Top 10 Brands with Positive and Negative Reviews")
plt.xlabel("Brands")
plt.ylabel("Percentage of Reviews")
plt.xticks(rotation=45)
plt.legend(title="Sentiment")
plt.tight_layout()
plt.show()


# Clorox has received the highest number of both positive (35%) and negative (30%) reviews. This is primarily because it makes up 35% of the total branded data, leading to a higher volume of feedback overall.

# ### Data Cleaning and Pre-processing
# 
# Based on the missing value analysis, we will decide on the appropriate strategy for handling missing values. This might involve imputation (replacing missing values with a calculated value like the mean, median, or mode) or removal (dropping rows or columns with missing values), depending on the extent and nature of the missing data.
# 
# We will also drop columns that are not relevant for our analysis to simplify the dataset and improve performance. Finally, we will ensure all columns have the correct data types for subsequent analysis.

# In[20]:


def clean_reviews_date(df, col='reviews_date'):
    """
    Cleans and standardizes a reviews_date column.

    Steps:
    1. Replace junk values (N/A, null, etc.)
    2. Parse valid dates with pandas (fast)
    3. For still-missing values, try dateutil parser (slower but flexible)
    4. Return cleaned dataframe and log summary
    """
    # Step 1: Replace common junk values with NA
    junk_values = ['N/A', 'NA', 'na', 'null', 'None', 'NONE', 'Unknown', '', ' ']
    df[col] = df[col].replace(junk_values, pd.NA)

    # Step 2: First attempt with pandas (fast, flexible)
    parsed = pd.to_datetime(df[col], errors='coerce')

    # Step 3: For rows still missing, try dateutil parser
    mask_missing = parsed.isna() & df[col].notna()
    if mask_missing.sum() > 0:
        def try_parse_date(x):
            try:
                return parser.parse(x, dayfirst=False, fuzzy=True)
            except:
                return pd.NaT
        parsed.loc[mask_missing] = df.loc[mask_missing, col].apply(try_parse_date)

    # Step 4: Assign cleaned column back
    df[col] = parsed

    # Step 5: Log summary
    total = len(df)
    valid = df[col].notna().sum()
    missing = df[col].isna().sum()
    print(f"[INFO] Cleaned '{col}': {valid}/{total} valid dates, {missing} missing ({missing/total:.2%})")

    return df


# In[21]:


# Example of handling missing values (replace with appropriate strategy based on analysis)

print("\nDropping irrelevant columns...")
df = df.drop(columns=['manufacturer'])   # redundant

columns_to_drop = ['reviews_userCity', 'reviews_userProvince']
df.drop(columns=columns_to_drop, inplace=True)
print(f"Dropped columns: {columns_to_drop}")

#Use user_sentiment or reviews_rating to impute:
#If sentiment = Positive (or rating ≥4) → fill "Yes".
#If sentiment = Negative (or rating ≤2) → fill "No".
#Neutral cases → "Unknown".
#This keeps imputation consistent with actual review content.

df['reviews_doRecommend'] = df.apply(
    lambda x: 'Yes' if pd.isna(x['reviews_doRecommend']) and x['reviews_rating'] >= 4
    else ('No' if pd.isna(x['reviews_doRecommend']) and x['reviews_rating'] <= 2
    else (x['reviews_doRecommend'] if pd.notna(x['reviews_doRecommend']) else 'Unknown')),
    axis=1
)

# fill missing with empty string
df['reviews_title'] = df['reviews_title'].fillna("")

# fill missing with Anonymous string
df['reviews_username'] = df['reviews_username'].fillna("Anonymous")


#convert data type
df = clean_reviews_date(df, 'reviews_date')  # convert to datetime
df['reviews_rating'] = df['reviews_rating'].astype('int')  # ensure integer
df['reviews_didPurchase'] = df['reviews_didPurchase'].fillna('Unknown').astype('category')
df['reviews_doRecommend'] = df['reviews_doRecommend'].astype('category')
df['user_sentiment'] = df['user_sentiment'].astype('category')

# Display the first few rows of the cleaned and pre-processed DataFrame
print("\nCleaned and Pre-processed DataFrame head:")
display(df.head())


# In[22]:


# reviews_date (54 missing, <0.2%)
# Very few → you don’t lose much if you drop them.
# Since date is important for time-based analysis but not for core sentiment/recommendation we can drop those rows

df = df[df['reviews_date'].notna()]

df = df[df['user_sentiment'].notna()]


# In[23]:


# Display the DataFrame info after dropping columns
print("\nDataFrame info after dropping columns:")
display(df.info())

# After handling missing values, display the updated missing value count
print("Missing values per column after handling:")
display(df.isnull().sum())


# In[24]:


# Download necessary NLTK data (if not already downloaded)
try:
    stopwords = stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    stopwords = stopwords.words('english')

try:
    WordNetLemmatizer()
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """
    Cleans and preprocesses text data.

    Steps:
    1. Convert text to lowercase.
    2. Remove punctuation.
    3. Remove stop words.
    4. Apply stemming or lemmatization (choose one).
    """
    # 1. Convert text to lowercase
    text = text.lower()

    # 2. Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # 3. Remove stop words
    text = ' '.join([word for word in text.split() if word not in stopwords])

    # 4. Apply lemmatization (you can switch to stemming if preferred)
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    # text = ' '.join([stemmer.stem(word) for word in text.split()]) # Uncomment for stemming

    return text

# Apply preprocessing to the reviews_text and reviews_title columns
print("Applying text preprocessing to 'reviews_text' and 'reviews_title'...")
df['reviews_text_preprocessed'] = df['reviews_text'].apply(preprocess_text)
df['reviews_title_preprocessed'] = df['reviews_title'].apply(preprocess_text)

# Display the first few rows with the new preprocessed columns
print("\nDataFrame head with preprocessed text:")
display(df[['reviews_text', 'reviews_text_preprocessed', 'reviews_title', 'reviews_title_preprocessed']].head())


# In[25]:


#dataset_overview(df)


# In[26]:


from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
# Set of stopwords for WordCloud
custom_stopwords = set(STOPWORDS)

# Generate WordCloud from lemmatized reviews
wordcloud = WordCloud(
    max_words=30,
    max_font_size=60,
    background_color='white',
    stopwords=custom_stopwords,
    random_state=42
).generate(" ".join(df['reviews_text_preprocessed']))

# Plot the WordCloud
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Top 30 Frequent Words in Reviews", fontsize=14)
plt.show()


# In[27]:


# Saving the individual models in a file
save_model(df, "df_preprocessed")
# # Later, load them by name
# loaded_cleansed_data = load_model("cleansed_df.pkl")


# ## 1.2 Train-Test Split

# In[28]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE

# Step 1: Split data into training and testing parts
# We will use 'reviews_text_preprocessed' and 'user_sentiment' for our model
X = df['reviews_text_preprocessed']
y = df['user_sentiment']

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Data split into training and testing sets.")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")



# ## Model Building & Evaluation

# #### Evaluation util function

# In[29]:


def evaluate_train_test_performance(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    positive_label="Positive",
    model_name="Model"
):
    """
    Evaluation with 4 horizontal subplots:
    1. Classification report (text)
    2. Confusion matrix
    3. ROC curve
    4. Summary metrics table (text)
    """

    # ---------- Helpers ----------
    def get_scores(m, X):
        if hasattr(m, "predict_proba"):
            pos_idx = list(m.classes_).index(positive_label)
            return m.predict_proba(X)[:, pos_idx]
        return None

    def calc_metrics(y_true, y_pred, y_score=None):
        final_est = model[-1] if hasattr(model, "__getitem__") else model
        labels = list(final_est.classes_)

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        tn, fp = cm[0]
        fn, tp = cm[1]

        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0),
            "Recall": recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0),
            "F1 Score": f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0),
            "Specificity": tn / (tn + fp) if (tn + fp) else 0.0
        }

        if y_score is not None:
            metrics["ROC AUC"] = roc_auc_score(
                (y_true == positive_label).astype(int),
                y_score
            )

        return cm, metrics

    # ---------- Predictions ----------
    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)

    y_train_score = get_scores(model, X_train)
    y_test_score  = get_scores(model, X_test)

    cm_train, m_train = calc_metrics(y_train, y_train_pred, y_train_score)
    cm_test,  m_test  = calc_metrics(y_test,  y_test_pred,  y_test_score)

    # ---------- Prepare text ----------
    cls_report = classification_report(y_test, y_test_pred, zero_division=0)

    summary_df = pd.DataFrame({"Train": m_train, "Test": m_test})
    summary_df = (summary_df * 100).round(2)

    # ---------- Figure ----------
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(f"{model_name} — Model Evaluation", fontsize=14, y=1.05)

    # === 1️⃣ Classification report ===
    axes[0].axis("off")
    axes[0].set_title("Classification Report (Test)", fontsize=11)
    axes[0].text(
        0,
        1,
        cls_report,
        ha="left",
        va="top",
        family="monospace",
        fontsize=9
    )

    # === 2️⃣ Confusion Matrix ===
    cm_norm = cm_test / cm_test.sum(axis=1, keepdims=True)

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar=False,
        xticklabels=model.classes_,
        yticklabels=model.classes_,
        ax=axes[1]
    )

    for i in range(2):
        for j in range(2):
            axes[1].text(
                j + 0.5,
                i + 0.75,
                f"\n({cm_test[i, j]})",
                ha="center",
                va="center",
                fontsize=9
            )

    axes[1].set_title("Confusion Matrix (Test)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    # === 3️⃣ ROC Curve ===
    if y_test_score is not None:
        fpr_tr, tpr_tr, _ = roc_curve(
            (y_train == positive_label).astype(int),
            y_train_score
        )
        fpr_te, tpr_te, _ = roc_curve(
            (y_test == positive_label).astype(int),
            y_test_score
        )

        axes[2].plot(fpr_tr, tpr_tr, lw=1.5, label=f"Train AUC = {m_train['ROC AUC']:.3f}")
        axes[2].plot(fpr_te, tpr_te, lw=2, label=f"Test AUC = {m_test['ROC AUC']:.3f}")
        axes[2].plot([0, 1], [0, 1], "--", color="gray")

        axes[2].set_title("ROC Curve")
        axes[2].set_xlabel("False Positive Rate")
        axes[2].set_ylabel("True Positive Rate")
        axes[2].legend(loc="lower right")
    else:
        axes[2].axis("off")
        axes[2].text(0.5, 0.5, "ROC not available", ha="center", va="center")

    # === 4️⃣ Summary Metrics ===
    axes[3].axis("off")
    axes[3].set_title("Summary Metrics (%)", fontsize=11)

    summary_text = "\n".join(
        f"{idx:<12}  Train: {row['Train']:>6.2f}%   Test: {row['Test']:>6.2f}%"
        for idx, row in summary_df.iterrows()
    )

    axes[3].text(
        0,
        1,
        summary_text,
        ha="left",
        va="top",
        fontsize=10,
        family="monospace"
    )

    plt.tight_layout()
    plt.show()

    return {"train": m_train, "test": m_test}


# In[30]:


def evaluate_models(
    models,
    X_train,
    y_train,
    X_test,
    y_test,
    positive_label="Positive"
):
    """
    Evaluates multiple trained models provided as a dictionary:
    {model_name: model_object}

    Returns:
        all_metrics: dict of metrics for each model
    """

    all_metrics = {}

    for model_name, model in models.items():
        print("\n" + "=" * 80)
        print(f"🔍 Evaluating {model_name}")
        print("=" * 80)

        metrics = evaluate_train_test_performance(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            positive_label=positive_label,
            model_name=model_name
        )

        all_metrics[model_name] = metrics

    return all_metrics


# ### Claasic model (Base Model) Pipelines 

# In[74]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from imblearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
# Step 3: Build and train models

tfidf = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=50_000
)

log_reg = Pipeline([
        ("tfidf", tfidf),
        ("smote", SMOTE(random_state=42)),
        ("clf", LogisticRegression(max_iter=1000))
    ])

rf_clf = Pipeline([
        ("tfidf", tfidf),
        ("smote", SMOTE(random_state=42)),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ])
nb_clf = Pipeline([
        ("tfidf", tfidf),
        ("smote", SMOTE(random_state=42)),
        ("clf", MultinomialNB())
    ])

models = {
    "Logistic Regression": log_reg,
    "Random Forest": rf_clf,
    "Naive Bayes": nb_clf
}


# In[75]:


# Train each model
print("Training models...")
for name, model in tqdm(models.items(), desc="Models"):
    model.fit(X_train, y_train)

print("\nAll models trained.")


# In[76]:


#model.fit(X_train_resampled, y_train_resampled)   X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)

## ─── 2️⃣ Train on the oversampled training data ───
#logreg_clf.fit(X_train_bal, y_train_bal)

all_model_metrics = evaluate_models(
    models=models,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    positive_label="Positive"
)




# ### XGBoost Classifier (Base Model)

# In[77]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import numpy as np


# Encode labels
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)

neg, pos = np.bincount(y_train_enc)
scale_pos_weight = neg / pos

print("scale_pos_weight:", scale_pos_weight)


xgb_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=30000,
        ngram_range=(1, 2)
    )),
    ("clf", xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,  # 🔥 imbalance handling
        eval_metric="logloss",
        tree_method="hist",                 # fast + stable on CPU
        random_state=42,
        n_jobs=-1
    ))
])


xgb_pipeline.fit(X_train, y_train_enc)


# In[78]:


xgb_model_name = "XGBoost"
print("\n" + "=" * 80)
print(f"🔍 Evaluating {xgb_model_name}")
print("=" * 80)

metrics_xgb = evaluate_train_test_performance(
    model=xgb_pipeline,
    X_train=X_train,
    y_train=y_train_enc,
    X_test=X_test,
    y_test=y_test_enc,
    positive_label=1,                 # 🔑 numeric
    model_name=xgb_model_name
)

all_model_metrics["xgb_model_name"] = metrics_xgb


# #### Logistic Regression (Fine Tune Model)

# In[79]:


from scipy.stats import loguniform
param_space = {
    "tfidf__max_features": [20_000, 40_000],
    "tfidf__ngram_range": [(1, 1), (1, 2)],

    "clf__C": loguniform(1e-2, 10),
    "clf__solver": ["liblinear"],
    "clf__class_weight": [None, "balanced"],
    "clf__tol": [1e-4],
    "clf__max_iter": [200]
}


cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

f1_custom = make_scorer(
    f1_score,
    pos_label="Positive",
    zero_division=0
)

logreg_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english"
    )),
    ("clf", LogisticRegression(
        random_state=42
    ))
])

search = RandomizedSearchCV(
    estimator=logreg_pipe,
    param_distributions=param_space,
    n_iter=10,              # ⏱️ fast
    scoring=f1_custom,
    cv=cv,
    verbose=1,
    random_state=42,
    refit=True
)
search.fit(X_train, y_train)


# In[80]:


print("RandomizedSearchCV completed.")
print("Best Params:")
print(search.best_params_)

best_logreg = search.best_estimator_

metrics_lr_tuned = evaluate_train_test_performance(
    model=best_logreg,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    positive_label="Positive",
    model_name="Logistic Regression (Tuned, Pipeline)"
)

all_model_metrics["Logistic Regression Tuned"] = metrics_lr_tuned


# #### Random Forest Classifier (Fine Tuning Model)

# In[81]:


# ─── Imports ───
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score

# ─── Custom F1 scorer for string labels ───
f1_custom = make_scorer(f1_score, pos_label="Positive", zero_division=0)

# ─── Step 1: Stratified train/validation split ───
X_trn, X_val, y_trn, y_val = train_test_split(
    X_train, y_train,
    test_size=0.2,
    stratify=y_train,
    random_state=42
)

# ─── Step 2: Pipeline definition ───
rf_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=30_000)),
    ("clf", RandomForestClassifier(random_state=42, n_jobs=1, class_weight="balanced"))
])

# ─── Step 3: Parameter grid ───
param_grid = {
    "clf__n_estimators":     [100, 200],
    "clf__max_depth":        [None, 10],
    "clf__min_samples_leaf": [1, 2],
    "clf__max_features":     ["sqrt"]
}

# ─── Step 4: Halving Grid Search ───
halving_search = HalvingGridSearchCV(
    estimator=rf_pipe,
    param_grid=param_grid,
    scoring=f1_custom,
    cv=3,
    factor=2,
    resource="n_samples",
    verbose=1,
    n_jobs=1
)

# ─── Step 5: Fit model ───
halving_search.fit(X_trn, y_trn)


# In[82]:


print("Random Forest completed.")
print("Best Params:")
print(search.best_params_)

# ─── Step 6: Best estimator ───
best_rf = halving_search.best_estimator_
print("🏅 Best RF Params:", halving_search.best_params_)

# ─── Step 7: Evaluate train & validation ───
metrics_random_forest_tuned = evaluate_train_test_performance(
    model=best_rf,
    X_train=X_trn, y_train=y_trn,
    X_test=X_val, y_test=y_val,
    positive_label="Positive",
    model_name="Random Forest (Tuned, Pipeline)"
)

all_model_metrics["Random Forest Tuned"] = metrics_random_forest_tuned


# #### XGBoost Classifier (Fine Tune Model)

# In[83]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
import numpy as np

# ---------------- 1️⃣ Encode labels ----------------
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)   # y_train from your original split
y_test_enc  = le.transform(y_test)

# ---------------- 2️⃣ Stratified train/validation split ----------------
X_trn, X_val, y_trn, y_val = train_test_split(
    X_train, y_train_enc,
    test_size=0.2,
    stratify=y_train_enc,
    random_state=42
)

# ---------------- 3️⃣ Compute scale_pos_weight for XGBoost ----------------
neg = np.sum(y_trn == 0)
pos = np.sum(y_trn == 1)
scale_pw = neg / pos if pos else 1.0

# ---------------- 4️⃣ Pipeline: TF-IDF + SMOTE + XGB ----------------
xgb_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=30_000)),
    ("smote", SMOTE(random_state=42)),
    ("xgb", XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pw,
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])


# ---------------- 5️⃣ GridSearchCV ----------------
param_grid = {
    "xgb__max_depth": [3],
    "xgb__learning_rate": [0.1],
    "xgb__n_estimators": [50, 100]
}

f1_scorer_custom = make_scorer(f1_score, pos_label=1)

cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=xgb_pipeline,
    param_grid=param_grid,
    scoring=f1_scorer_custom,
    cv=cv,
    verbose=1,
    n_jobs=1
)

# ---------------- 6️⃣ Fit GridSearchCV ----------------
grid_search.fit(X_trn, y_trn)


# In[84]:


best_xgb_pipeline = grid_search.best_estimator_
print("Best Params:", grid_search.best_params_)

# ---------------- 7️⃣ Evaluate ----------------
metrics_xgb_tuned = evaluate_train_test_performance(
    model=best_xgb_pipeline,
    X_train=X_trn, y_train=y_trn,
    X_test=X_val,  y_test=y_val,
    positive_label=1,
    model_name="XGBoost (Pipeline Tuned)"
)

all_model_metrics["XGBoost Tuned"] = metrics_xgb_tuned


# In[85]:


# List to hold summary for each model
summary_list = []

for model_name, metrics in all_model_metrics.items():
    # metrics is a dict: {"train": {...}, "test": {...}}
    train_metrics = metrics["train"]
    test_metrics  = metrics["test"]
    
    # Create a DataFrame row
    df_row = pd.DataFrame({
        "Model": [model_name]*len(train_metrics),
        "Metric": list(train_metrics.keys()),
        "Train": list(train_metrics.values()),
        "Test": list(test_metrics.values())
    })
    
    summary_list.append(df_row)

# Concatenate all rows
summary_df = pd.concat(summary_list, ignore_index=True)

# Optionally, format percentages for easier reading
summary_df_formatted = summary_df.copy()
for col in ["Train", "Test"]:
    summary_df_formatted[col] = (summary_df_formatted[col]*100).round(2)

# Display
summary_df_formatted


# ### Model Performance Comparison
# 
# #### Naive Bayes
# Naive Bayes is the weakest-performing model among the three, recording the lowest scores across all evaluation metrics. This indicates that it is less effective for this classification task compared to Logistic Regression and Random Forest.
# 
# ---
# 
# #### Logistic Regression vs Random Forest
# 
# - **Accuracy:**  
#   Random Forest slightly outperforms Logistic Regression (0.928 vs 0.921).
# 
# - **Precision:**  
#   Logistic Regression achieves higher precision (0.980 vs 0.950), indicating fewer false positives.
# 
# - **Recall:**  
#   Random Forest significantly outperforms Logistic Regression (0.969 vs 0.929), meaning it captures a larger proportion of positive instances.
# 
# - **F1-Score:**  
#   Random Forest attains a higher F1-score (0.960 vs 0.954), reflecting a better balance between precision and recall.
# 
# - **ROC-AUC:**  
#   Logistic Regression demonstrates stronger overall class-separation capability (0.961 vs 0.946).
# 
# ---
# 
# #### Model Selection Insight
# 
# ##### Best Overall Accuracy & F1 Score
# - **Tuned Logistic Regression** has excellent recall (99.93%) and F1 score (99.68%), making it the most balanced model.
# - **Random Forest** shows very high training accuracy (≈100%) but some drop on test data, especially in specificity.
# 
# ##### Naive Bayes
# - Consistent performance but slightly lower recall than Logistic Regression and Random Forest.
# 
# ##### XGBoost
# - **Base XGBoost** is reasonable (F1 ~94%).
# - **Tuned XGBoost** shows very high precision but low recall (~52%), causing low overall accuracy (~57%).
# - Indicates tuning favors precision over recall — class weighting or early stopping might need adjustment.
# 
# ##### Specificity vs Recall Tradeoff
# - Models like Tuned XGBoost prioritize precision/specificity, which lowers F1 and overall accuracy.
# 
# ##### Actionable Notes
# - **Logistic Regression Tuned** is the safest, most balanced choice.
# - **Random Forest Tuned** is strong but slightly inconsistent in specificity.
# - **XGBoost** needs careful tuning of class balance and early stopping to improve recall.
# 

# #### Save best performing model

# In[86]:


save_model(best_logreg, "logistic-regression-tuned")


# ## Build Recomendation System
# 

# We will build two types of recommendation systems to evaluate which performs better for our use case:
# 
# ##### User-Based Recommendation System
#     Recommends products based on the preferences of similar users.
# 
# ##### Item-Based Recommendation System
#     Recommends products based on similarities between items.
# 
# Once both models are developed, we will analyze their performance and select the approach that best fits our scenario.
# After identifying the best-performing model, we will proceed to the core task:
# 
# ***Recommending the top 20 products that a user is most likely to purchase based on their past ratings.***
# 

# #### Prepare data for recommendation system
# 
# Select the necessary columns and potentially preprocess them for building the recommendation system.
# 

# **Reasoning**:
# Create a new DataFrame from previously processed dataframe(**df**) with only the necessary columns for the recommendation system and display its head and info.
# 
# 

# In[87]:


reviews_df = df[['reviews_username', 'name', 'reviews_rating']]


# In[88]:


dataset_overview(reviews_df)


# #### Split data
# 
# Divide the data into training and testing sets for evaluating the recommendation system.
# 

# **Reasoning**:
# The goal is to split the data into training and testing sets for the recommendation system. The previous subtask created the `reviews_df` DataFrame with relevant columns. This step will perform the split using `train_test_split`.
# 
# 

# In[89]:


from sklearn.model_selection import train_test_split

# Split the reviews_df DataFrame into training and testing sets
train_df, test_df = train_test_split(reviews_df, test_size=0.2, random_state=42)

# Print the shapes of the resulting DataFrames
print("Training set shape:", train_df.shape)
print("Testing set shape:", test_df.shape)


# ### Build user-based collaborative filtering model
# 
# Implement a user-based collaborative filtering recommendation system.
# 

# **Reasoning**:
# Create a pivot table from the training data, calculate user similarity, and define a function for user-based recommendations.
# 
# 

# In[90]:


from sklearn.metrics.pairwise import cosine_similarity

# 1. Create a pivot table from the training data
user_item_matrix = train_df.pivot_table(index='reviews_username', columns='name', values='reviews_rating').fillna(0)

print("User-Item Matrix head:")
display(user_item_matrix.head())
print("\nUser-Item Matrix shape:", user_item_matrix.shape)

# 2. Calculate the pairwise cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

print("\nUser Similarity Matrix head:")
display(user_similarity_df.head())



# #### Prediction - User-User

# In[91]:


# 3. Define a function for user-based recommendations
def user_based_recommendations(user_id, user_item_matrix, user_similarity_df, n_recommendations=5):
    """
    Generates user-based recommendations for a given user.

    Args:
        user_id (str): The ID of the target user.
        user_item_matrix (pd.DataFrame): The user-item matrix.
        user_similarity_df (pd.DataFrame): The user similarity matrix.
        n_recommendations (int): The number of recommendations to generate.

    Returns:
        list: A list of recommended item IDs.
    """
    if user_id not in user_similarity_df.index:
        print(f"User '{user_id}' not found in the similarity matrix.")
        return []

    # Get the similarity scores for the target user
    user_similarities = user_similarity_df.loc[user_id]

    # Remove the user's own similarity score
    user_similarities = user_similarities.drop(user_id)

    # Sort similar users by similarity in descending order
    similar_users = user_similarities.sort_values(ascending=False)

    # Get items rated by the target user
    items_rated_by_user = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index

    # Initialize a dictionary to store recommended item scores
    item_scores = {}

    # Iterate through similar users
    for similar_user, similarity_score in similar_users.items():
        if similarity_score <= 0: # Consider only users with positive similarity
            continue

        # Get items rated by the similar user
        items_rated_by_similar_user = user_item_matrix.loc[similar_user][user_item_matrix.loc[similar_user] > 0].index

        # Identify items rated by the similar user but not by the target user
        items_to_consider = items_rated_by_similar_user.difference(items_rated_by_user)

        # For each item, add the similar user's rating weighted by similarity
        for item in items_to_consider:
            if item not in item_scores:
                item_scores[item] = 0
            item_scores[item] += user_item_matrix.loc[similar_user, item] * similarity_score

    # Sort items by their recommendation score in descending order
    recommended_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)

    # Return the top N recommended item IDs
    return [item for item, score in recommended_items[:n_recommendations]]

print("\nUser-based recommendation function defined.")


# ### Build item-based collaborative filtering model
# 
# Implement an item-based collaborative filtering recommendation system.
# 

# **Reasoning**:
# Calculate item similarity and define the item-based recommendation function.
# 
# 

# In[92]:


from sklearn.metrics.pairwise import cosine_similarity

# 1. Calculate the pairwise cosine similarity between items (using the transposed user-item matrix)
item_similarity = cosine_similarity(user_item_matrix.T)

# 2. Convert the resulting similarity matrix into a pandas DataFrame
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

print("Item Similarity Matrix head:")
display(item_similarity_df.head())
print("\nItem Similarity Matrix shape:", item_similarity_df.shape)



# #### Prediction - Item Item

# In[103]:


# 3. Define a function item_based_recommendations
def item_based_recommendations(item_name, user_item_matrix, item_similarity_df, n_recommendations=5):
    """
    Generates item-based recommendations for a given item.

    Args:
        item_name (str): The name of the target item.
        user_item_matrix (pd.DataFrame): The user-item matrix.
        item_similarity_df (pd.DataFrame): The item similarity matrix.
        n_recommendations (int): The number of recommendations to generate.

    Returns:
        list: A list of recommended item names.
    """
    if item_name not in item_similarity_df.index:
        print(f"Item '{item_name}' not found in the similarity matrix.")
        return []

    # 4. Get the similarity scores for the target item
    item_similarities = item_similarity_df.loc[item_name]

    # 5. Remove the item's own similarity score
    item_similarities = item_similarities.drop(item_name, errors='ignore')

    # 6. Sort similar items by similarity in descending order
    similar_items = item_similarities.sort_values(ascending=False)

    # 7. Get users who rated the target item
    users_who_rated_item = user_item_matrix.index[user_item_matrix[item_name] > 0]

    # 8. Initialize a dictionary to store recommended item scores
    item_scores = {}

    # 9. Iterate through similar items and users
    for similar_item, similarity_score in similar_items.items():
        if similarity_score <= 0: # Consider only items with positive similarity
            continue

        # 10. For each user, if they rated the similar item, add their rating weighted by the item similarity
        for user in users_who_rated_item:
            if user_item_matrix.loc[user, similar_item] > 0:
                if similar_item not in item_scores:
                    item_scores[similar_item] = 0
                item_scores[similar_item] += user_item_matrix.loc[user, similar_item] * similarity_score

    # 11. Sort items in the item_scores dictionary by their recommendation score
    recommended_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)

    # 12. Return the top N recommended item names
    return [item for item, score in recommended_items[:n_recommendations]]

print("\nItem-based recommendation function defined.")


# ### Evaluate recommendation systems
# 
# Evaluate the performance of both the user-based and item-based recommendation systems using appropriate metrics (e.g., RMSE, precision, recall).
# 

# **Reasoning**:
# Define the evaluation function for the recommendation systems and then call it for both user-based and item-based models.
# 
# 

# In[93]:


def evaluate_recommendation_system(test_df, recommendation_function, user_item_matrix, similarity_matrix, n_recommendations=5):
    """
    Evaluates the performance of a recommendation system.

    Args:
        test_df (pd.DataFrame): DataFrame containing the test set (user, item, rating).
        recommendation_function (function): The function to generate recommendations.
        user_item_matrix (pd.DataFrame): The user-item matrix from the training data.
        similarity_matrix (pd.DataFrame): The user or item similarity matrix.
        n_recommendations (int): The number of recommendations generated by the system.

    Returns:
        dict: A dictionary containing evaluation metrics (e.g., hit rate).
    """
    hits = 0
    total_test_interactions = len(test_df)
    users_in_train = user_item_matrix.index.tolist()
    items_in_train = user_item_matrix.columns.tolist()

    for index, row in tqdm(test_df.iterrows(), total=total_test_interactions, desc=f"Evaluating {recommendation_function.__name__}"):
        user = row['reviews_username']
        actual_item = row['name']

        # Only evaluate if the user and item are in the training data's matrix
        # This is a limitation of collaborative filtering - it cannot recommend for new users/items
        if user in users_in_train and actual_item in items_in_train:
            # Generate recommendations for the user
            if recommendation_function.__name__ == 'user_based_recommendations':
                 recommended_items = recommendation_function(user, user_item_matrix, similarity_matrix, n_recommendations)
            elif recommendation_function.__name__ == 'item_based_recommendations':
                 # For item-based, we need to provide an item from the user's history in the training set
                 # This is a simplification; a real system would use all items rated by the user
                 user_rated_items_in_train = user_item_matrix.loc[user][user_item_matrix.loc[user] > 0].index.tolist()
                 if not user_rated_items_in_train: # Skip if user has no rated items in training (should be handled by user_in_train check, but double-checking)
                     continue
                 # Use the first item the user rated in the training set as a basis for item-based recommendation
                 # A more sophisticated approach would aggregate recommendations from all rated items
                 seed_item = user_rated_items_in_train[0]
                 recommended_items = recommendation_function(seed_item, user_item_matrix, similarity_matrix, n_recommendations)
            else:
                print(f"Unknown recommendation function: {recommendation_function.__name__}")
                continue


            # Check if the actual item is in the recommendations
            if actual_item in recommended_items:
                hits += 1

    # Calculate metrics
    hit_rate = hits / total_test_interactions if total_test_interactions > 0 else 0

    return {"Hit Rate": hit_rate}



# In[94]:


# Evaluate User-Based Recommendation System
print("Evaluating User-Based Recommendation System...")
user_based_metrics = evaluate_recommendation_system(test_df, user_based_recommendations, user_item_matrix, user_similarity_df)
print("User-Based Metrics:", user_based_metrics)

# Evaluate Item-Based Recommendation System
print("\nEvaluating Item-Based Recommendation System...")
item_based_metrics = evaluate_recommendation_system(test_df, item_based_recommendations, user_item_matrix, item_similarity_df)
print("Item-Based Metrics:", item_based_metrics)


# **Reasoning**:
# The evaluation of the item-based recommendation system is complete, print the metrics.
# 
# 

# In[95]:


print("Item-Based Metrics:", item_based_metrics)


# ### Compare and select the best model
# 
# Compare the performance of the two recommendation systems and select the best one based on the evaluation results, providing reasons for the selection.
# 

# **Reasoning**:
# Compare the evaluation metrics and select the best performing model.
# 
# 

# In[96]:


print("User-Based Metrics:", user_based_metrics)
print("Item-Based Metrics:", item_based_metrics)

# Compare the Hit Rates and select the best model
if item_based_metrics['Hit Rate'] > user_based_metrics['Hit Rate']:
    best_model = "Item-Based Collaborative Filtering"
    best_metric_value = item_based_metrics['Hit Rate']
    reason = f"The Item-Based model achieved a higher Hit Rate ({best_metric_value:.4f}) compared to the User-Based model ({user_based_metrics['Hit Rate']:.4f})."
elif user_based_metrics['Hit Rate'] > item_based_metrics['Hit Rate']:
    best_model = "User-Based Collaborative Filtering"
    best_metric_value = user_based_metrics['Hit Rate']
    reason = f"The User-Based model achieved a higher Hit Rate ({best_metric_value:.4f}) compared to the Item-Based model ({item_based_metrics['Hit Rate']:.4f})."
else:
    best_model = "Both models performed equally"
    best_metric_value = user_based_metrics['Hit Rate'] # or item_based_metrics['Hit Rate']
    reason = f"Both models achieved the same Hit Rate ({best_metric_value:.4f})."

print(f"\nBest Performing Recommendation Model: {best_model}")
print(f"Reason for Selection: {reason}")


# #### Summary of recomendation model:
# 
# ##### Data Analysis Key Findings
# 
# *   The user-item matrix created from the training data contained 20527 unique users and 263 unique items.
# *   The user-based recommendation system achieved a Hit Rate of approximately 0.0770 on the test set.
# *   The item-based recommendation system achieved a Hit Rate of approximately 0.0828 on the test set, slightly outperforming the user-based model.
# 
# ##### Insights
# 
# *   The item-based collaborative filtering model is selected as the best performing model due to its higher Hit Rate on the test data.
# *   Further improvements could involve exploring different similarity metrics, incorporating regularization techniques, or utilizing hybrid approaches combining content-based or model-based methods to address the limitations of collaborative filtering (e.g., cold start problem).
# 

# In[97]:


# Step 8: Generate recommendations for a user using the best model

# Specify the username for whom you want recommendations
# Replace 'Enter_Username_Here' with the actual username from your dataset
target_username = 'rebecca' # Example username, replace with a user from your dataset

# Check if the target user exists in the user-item matrix
if target_username not in user_item_matrix.index:
    print(f"User '{target_username}' not found in the training data.")
else:
    # To use the item-based recommendation function, we need an item the user has rated
    # We can pick one of the items the user rated in the training set as a seed
    user_rated_items_in_train = user_item_matrix.loc[target_username][user_item_matrix.loc[target_username] > 0].index.tolist()

    if not user_rated_items_in_train:
        print(f"User '{target_username}' has not rated any items in the training data. Cannot generate item-based recommendations.")
    else:
        # Use the first item the user rated in the training set as the seed item
        seed_item_for_recommendation = user_rated_items_in_train[0]
        print(f"Using '{seed_item_for_recommendation}' as the seed item for recommendations for user '{target_username}'.")

        # Generate top N recommendations using the item-based model
        n_recommendations = 20
        recommended_items = item_based_recommendations(seed_item_for_recommendation, user_item_matrix, item_similarity_df, n_recommendations)

        if recommended_items:
            print(f"\nTop {n_recommendations} Recommended Items for user '{target_username}':")
            for i, item in enumerate(recommended_items):
                print(f"{i+1}. {item}")
        else:
            print(f"Could not generate recommendations for user '{target_username}'.")


# 
# Analyze the reviews of the top 20 recommended products for a user, predict the sentiment of these reviews using the best performing sentiment analysis model, calculate the percentage of positive sentiments for each of the 20 products, and identify the top 5 products with the highest percentage of positive reviews.

# ## Filter reviews for recommended products
# 
# Create a DataFrame containing only the reviews for the top 20 recommended products.
# 

# **Reasoning**:
# Create a DataFrame containing only the reviews for the top 20 recommended products.
# 
# 

# In[98]:


# 1. Create a list named top_20_recommended_items
top_20_recommended_items = recommended_items

# 2. Filter the original DataFrame df
df_recommended_reviews = df[df['name'].isin(top_20_recommended_items)]

# 3. Display the head and the shape of the df_recommended_reviews DataFrame
print("DataFrame head with reviews for top 20 recommended products:")
display(df_recommended_reviews.head())
print("\nShape of the filtered DataFrame:", df_recommended_reviews.shape)


# ## Improving the recommendations using the sentiment analysis model
# 
# Use the best sentiment analysis model (Random Forest, based on previous evaluation) to predict the sentiment (positive or negative) for each review of the recommended products.
# 

# **Reasoning**:
# Use the best sentiment analysis model to predict the sentiment of the reviews for the recommended products.
# 
# 

# In[108]:


# 1. Select the 'reviews_text_preprocessed' column from the df_recommended_reviews DataFrame
X_recommended = df_recommended_reviews['reviews_text_preprocessed']

# 3. Use the best performing sentiment analysis model (Logistic Regression Tuned) to predict the sentiment labels
# The best model was identified as Logistic Regression Tunedin the previous sentiment analysis task.
predicted_sentiment = best_logreg.predict(X_recommended)

# 4. Add the predicted sentiment labels as a new column named 'predicted_sentiment' to the df_recommended_reviews DataFrame
df_recommended_reviews['predicted_sentiment'] = predicted_sentiment

# 5. Display the head of the df_recommended_reviews DataFrame
print("\nDataFrame head with predicted sentiment:")
display(df_recommended_reviews[['reviews_text', 'reviews_text_preprocessed', 'predicted_sentiment']].head())


# **Reasoning**:
# Calculate the percentage of positive sentiments for each recommended product and identify the top 5 products with the highest percentage.
# 
# 

# In[106]:


# Calculate the percentage of positive sentiments for each product
positive_sentiment_percentage = df_recommended_reviews.groupby('name')['predicted_sentiment'].apply(lambda x: (x == 'Positive').sum() / len(x) * 100)

# Sort the products by the percentage of positive sentiment in descending order
sorted_positive_sentiment = positive_sentiment_percentage.sort_values(ascending=False)

# Identify the top 5 products with the highest percentage of positive reviews
top_5_positive_products = sorted_positive_sentiment.head(5)

# Display the percentage of positive sentiment for all recommended products (sorted)
print("\nPercentage of Positive Sentiment for Recommended Products (Sorted):")
display(sorted_positive_sentiment)

# Display the top 5 products with the highest percentage of positive reviews
print("\nTop 5 Products with Highest Percentage of Positive Reviews:")
display(top_5_positive_products)


# ## Present the top 5 products
# 
# Present the names of the top 5 products with the highest percentage of positive reviews.
# 

# **Reasoning**:
# Print the heading and iterate through the top 5 positive products to display their rank, name, and positive sentiment percentage.
# 
# 

# In[109]:


# Print a clear heading
print("Top 5 Products with Highest Percentage of Positive Reviews:")

# Iterate through the top_5_positive_products Series and print the results
for rank, (product_name, percentage) in enumerate(top_5_positive_products.items()):
    print(f"{rank + 1}. {product_name}: {percentage:.2f}% Positive")


# #### Summary
# 
# ##### Data Analysis Key Findings
# 
# - A total of *11,924 reviews* were analyzed across the *top 20 recommended products*.
# - A *Random Forest model* was used to predict review sentiment, classifying each review as *Positive* or *Negative*.
# - The proportion of positive sentiment varies across products.
# - *My Big Fat Greek Wedding 2 (Blu-Ray + DVD + Digital)* has the highest percentage of positive reviews at *96.26%*.
# - The *top 5 products* with the highest percentage of positive reviews are:
#   1. *My Big Fat Greek Wedding 2 (Blu-Ray + DVD + Digital): 96.26% Positive*
#   2. *100: Complete First Season (Blu-Ray): 94.96% Positive*
#   3. *Planes: Fire & Rescue (2 Discs, includes Digital Copy) (Blu-Ray/DVD): 94.57% Positive*
#   4. *Clorox Disinfecting Bathroom Cleaner: 92.99% Positive*
#   5. *Godzilla 3D (Includes Digital Copy Ultraviolet, 3D/2D Blu-Ray/DVD): 92.84% Positive*
# 
# ##### Insights and Next Steps
# 
# - The high percentage of positive reviews for the top five products indicates strong customer satisfaction and suggests these items are well-suited for *featured placements, promotions, or recommendations*.
# - Additional analysis could focus on *negative reviews* from lower-performing products to identify recurring issues and potential areas for improvement.
# 
# 

# 
# ##### saving pickel files

# In[111]:


save_pickles = {
    "user_item_matrix": user_item_matrix,
    "item_similarity_df": item_similarity_df,
    "best_logreg": best_logreg,
    "df": df
}


# In[112]:


# Save each object
for name, obj in save_pickles.items():
    save_model(obj=obj, name=name)


# In[113]:


loaded = {}
for name in save_pickles.keys():
    loaded[name] = load_model(name=name)


# **Github Link For Repo :-** https://github.com/amitkmrjha/sentiment-based-recommendation-system

# **App Deployed on renderer Link :-**
# 
# https://product-recommendation-project-8a76b78d452d.herokuapp.com/

# In[ ]:




