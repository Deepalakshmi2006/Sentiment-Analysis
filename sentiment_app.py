# sentiment_app.py

import streamlit as st
import re
import joblib
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Clean user input
def clean_text(text):
    text = re.sub(r"http\S+|@\w+|#\w+|[^A-Za-z\s]", "", text)
    text = text.lower()
    return " ".join([word for word in text.split() if word not in stop_words])

# Streamlit UI
st.set_page_config(page_title="Tweet Sentiment Detection", layout="centered")
st.title("💬 Twitter Sentiment Detector")
st.write("Enter a tweet and find out if it's Positive or Negative.")

# Input text
tweet = st.text_area("✍️ Enter Tweet Text:")

# Predict
if st.button("Analyze Sentiment"):
    if tweet.strip() == "":
        st.warning("Please enter a tweet.")
    else:
        cleaned = clean_text(tweet)
        vectorized = vectorizer.transform([cleaned])
        result = model.predict(vectorized)[0]
        sentiment = "Positive 😊" if result == 1 else "Negative 😞"
        st.success(f"Sentiment: **{sentiment}**")
