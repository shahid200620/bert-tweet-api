import os
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "model_output")

app = FastAPI(
    title="Sentiment Analysis API",
    description="Sentiment analysis API using a fine-tuned DistilBERT model",
    version="1.0.0"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

class PredictionRequest(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: PredictionRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=96
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)[0]
    predicted_label = torch.argmax(probabilities).item()
    confidence = float(probabilities[predicted_label])

    sentiment = "positive" if predicted_label == 1 else "negative"

    return {
        "sentiment": sentiment,
        "confidence": confidence
    }