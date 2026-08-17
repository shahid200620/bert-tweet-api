import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "model_output"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

st.set_page_config(page_title="Twitter Sentiment Analysis", page_icon="🐦", layout="centered")

st.title("🐦 Twitter Sentiment Analysis")
st.write("Analyze the sentiment of a tweet using a fine-tuned DistilBERT model.")

text = st.text_area("Enter tweet text", placeholder="Example: I absolutely love this product!", height=150)

if st.button("Analyze Sentiment"):
    if not text.strip():
        st.error("Please enter some text.")
    else:
        tokenizer, model = load_model()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)

        labels = {0: "negative", 1: "positive"}
        sentiment = labels[prediction.item()]

        st.success(f"Sentiment: {sentiment.capitalize()}")
        st.metric("Confidence", f"{confidence.item() * 100:.2f}%")