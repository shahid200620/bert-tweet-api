import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

st.title("Sentiment Analysis")
st.write("Analyze the sentiment of a piece of text using our fine-tuned DistilBERT model.")

text = st.text_area(
    "Enter text",
    placeholder="Type a review, comment, or message here...",
    height=150
)

if st.button("Analyze Sentiment", use_container_width=True):
    if not text.strip():
        st.warning("Please enter some text before analyzing.")
    else:
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={"text": text},
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                sentiment = result["sentiment"]
                confidence = result["confidence"]

                st.subheader("Prediction")

                if sentiment == "positive":
                    st.success(f"Sentiment: {sentiment.upper()}")
                else:
                    st.error(f"Sentiment: {sentiment.upper()}")

                st.metric("Confidence", f"{confidence:.2%}")

                st.progress(confidence)
            else:
                st.error(f"API request failed: {response.text}")

        except requests.exceptions.RequestException:
            st.error("Unable to connect to the sentiment analysis API.")