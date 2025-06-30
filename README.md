# Twitter Sentiment Analysis

## Overview

**Twitter-sentiment-analysis-** is a system designed for live sentiment analysis of tweets in real time. It leverages modern NLP models to classify the sentiment of Twitter data as positive or negative, providing an interactive dashboard for data exploration. This solution is well-suited for data scientists and software engineers interested in natural language processing, data streaming, and real-time analytics.

---

## Features

- **Live Tweet Streaming:** Integrates with the Twitter API to stream tweets based on custom search queries.
- **Sentiment Analysis:** Fine-tunes and uses a DistilBERT-based model to classify tweet sentiment (positive/negative).
- **RAG-based Contextual Sentiment:** Uses retrieval-augmented generation (RAG) with FAISS and sentence transformers for context-aware sentiment predictions.
- **Interactive Visualization:** Streamlit dashboard displays sentiment distribution and recent tweet logs.
- **Data Export:** Download analyzed tweet data as CSV for further analysis.

---

## Getting Started

### 1. Environment Setup

Install required Python packages:

```bash
pip install streamlit tweepy transformers datasets sentence-transformers faiss-cpu torch numpy pandas
```

### 2. Dataset Preparation

- The core dataset used is [Sentiment140](https://www.kaggle.com/kazanova/sentiment140).
- The notebook `load_dataset.ipynb` guides you through:
  - Installing Kaggle API and authenticating.
  - Downloading and preprocessing the Sentiment140 dataset.
  - Cleaning columns and saving as `sentiment140_clean.csv`.

### 3. Model Fine-Tuning

- Use the notebook to fine-tune `distilbert-base-uncased` for sentiment classification.
- The model and tokenizer are saved to a directory (default: `distilbert_finetuned`).

### 4. Running the App

Start the Streamlit app:

```bash
streamlit run app.py
```

- Enter your Twitter API Bearer Token in `app.py` (`BEARER_TOKEN` variable).
- Use the dashboard to input search queries (e.g., "AI lang:en") and start streaming.

---

## Core Components

### `app.py`

- **Streamlit UI:** User inputs a search term and starts streaming.
- **Tweet Streaming:** Uses Tweepy to connect to Twitter's streaming API.
- **Sentiment Classification:** 
  - Cleans tweets (removes URLs, usernames, hashtags, punctuation).
  - Uses `distilbert_finetuned` for sentiment prediction.
  - Optionally augments with nearest-neighbor context using FAISS and MiniLM embeddings.
- **Visualization:** 
  - Realtime bar chart showing sentiment distribution.
  - Table of recent tweets and their sentiment.
  - Option to download results as CSV.

### `load_dataset.ipynb`

- **Data Download:** Automates Kaggle dataset download and extraction.
- **Data Cleaning:** Parses and exports a clean CSV for model training.
- **Model Fine-Tuning:** Shows how to train and save a custom DistilBERT sentiment classifier using HuggingFace Transformers.

### Requirements

- Python
- Twitter Developer Account (for API keys)
- Kaggle Account (for dataset access)

---

## Dependencies

- `streamlit` — Interactive dashboard.
- `tweepy` — Twitter API integration.
- `transformers`, `datasets` — HuggingFace ecosystem for NLP.
- `sentence-transformers`, `faiss-cpu` — For semantic search and context retrieval.
- `torch`, `numpy`, `pandas` — Core data and ML libraries.

---

## Example Usage

1. Prepare and fine-tune the model using `load_dataset.ipynb`.
2. Launch the app with Streamlit.
3. Enter a search term (e.g., "Elections lang:en"), click **Start Stream**.
4. Watch live sentiment trends and analyze/download results.

---

## Notes

- The app caches models and embeddings for performance.
- Make sure your Twitter API credentials have access to the streaming endpoint.
- The dashboard is highly extensible for additional analytics, models, or data sources.

---

## License

Copyright (c) Saronzeleke

Specify your license here (e.g., MIT, GPL).

---

## Contact

For questions or contributions, please open an issue or pull request on the GitHub repository.

Email: saronzeleke@gmail.com
