
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