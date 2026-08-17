import argparse
import os
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

def predict_text(text):
    response = requests.post(
        f"{API_URL}/predict",
        json={"text": text},
        timeout=60
    )
    response.raise_for_status()
    return response.json()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    df = pd.read_csv(args.input_file)

    if "text" not in df.columns:
        raise ValueError("Input CSV must contain a text column")

    sentiments = []
    confidences = []

    for text in df["text"].fillna(""):
        result = predict_text(str(text))
        sentiments.append(result["sentiment"])
        confidences.append(result["confidence"])

    df["predicted_sentiment"] = sentiments
    df["confidence"] = confidences

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(args.output_file, index=False)

    print(f"Processed {len(df)} rows")
    print(f"Created {args.output_file}")

if __name__ == "__main__":
    main()