
import streamlit as st
import tweepy
import torch
import re
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
from collections import defaultdict
from datetime import datetime
import time

BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAA..."
MODEL_DIR = "distilbert_finetuned"
DURATION = 300
MAX_STORAGE = 1000
EMBEDDING_DIM = 384
@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    return model, tokenizer

model, tokenizer = load_model()
encoder = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.IndexFlatL2(EMBEDDING_DIM)
tweets_storage = []
embeddings = []
def clean_tweet(tweet):
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'#\w+', '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    return tweet.strip()
def get_distilbert_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return {0: 'negative', 1: 'positive'}[predicted_class]
def rag_sentiment(tweet):
    cleaned_tweet = clean_tweet(tweet)
    embedding = encoder.encode([cleaned_tweet])[0]
    context = ""
    if len(tweets_storage) > 0:
        D, I = index.search(np.array([embedding]), k=3)
        context_tweets = [tweets_storage[i] for i in I[0] if i < len(tweets_storage)]
        context = " ".join(context_tweets)

    sentiment = get_distilbert_sentiment(cleaned_tweet)

    # Update storage and FAISS index
    tweets_storage.append(cleaned_tweet)
    embeddings.append(embedding)
    if len(tweets_storage) > MAX_STORAGE:
        tweets_storage.pop(0)
        embeddings.pop(0)
        index.reset()
        if embeddings:
            index.add(np.array(embeddings))
    else:
        index.add(np.array([embedding]))

    return sentiment

# --- Twitter Streaming Client ---
class TweetStream(tweepy.StreamingClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = time.time()
        self.tweet_count = 0

    def on_tweet(self, tweet):
        try:
            sentiment = rag_sentiment(tweet.text)
            st.session_state.sentiment_counts[sentiment] += 1
            st.session_state.tweet_log.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "tweet": tweet.text,
                "sentiment": sentiment
            })
            self.tweet_count += 1
            if time.time() - self.start_time > DURATION:
                self.disconnect()
        except Exception as e:
            pass

# --- Streamlit UI ---
st.set_page_config(page_title="Live Twitter Sentiment Dashboard", layout="wide")
st.title("🌍 Live Twitter Sentiment Analysis")

if 'sentiment_counts' not in st.session_state:
    st.session_state.sentiment_counts = defaultdict(int)
if 'tweet_log' not in st.session_state:
    st.session_state.tweet_log = []

search_query = st.text_input("Enter search term (e.g., 'AI', 'Ethiopia', 'Climate'):", value="AI lang:en")

start_button = st.button("Start Stream")

if start_button:
    st.write(f"Streaming for {DURATION // 60} minutes...")

    stream = TweetStream(BEARER_TOKEN)

    existing_rules = stream.get_rules()
    if existing_rules and existing_rules.data:
        rule_ids = [rule.id for rule in existing_rules.data]
        stream.delete_rules(rule_ids)

    stream.add_rules(tweepy.StreamRule(search_query))

    stream.filter(tweet_fields=["text"])

# --- Display Sentiment Chart ---
if st.session_state.sentiment_counts:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    labels = list(st.session_state.sentiment_counts.keys())
    values = list(st.session_state.sentiment_counts.values())
    ax.bar(labels, values, color=['red', 'green', 'gray'])
    ax.set_title("Sentiment Distribution")
    st.pyplot(fig)

# --- Display Tweets ---
if st.session_state.tweet_log:
    st.subheader("Recent Tweets Analyzed")
    df = pd.DataFrame(st.session_state.tweet_log)
    st.dataframe(df)

    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(df)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='sentiment_analysis.csv',
        mime='text/csv'
    )