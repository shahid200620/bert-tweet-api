import os
import re
import pandas as pd
from datasets import load_dataset

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^A-Za-z0-9\s!?.,']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    dataset = load_dataset("stanfordnlp/imdb")

    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])

    train_df["text"] = train_df["text"].apply(clean_text)
    test_df["text"] = test_df["text"].apply(clean_text)

    train_df = train_df[["text", "label"]]
    test_df = test_df[["text", "label"]]

    train_df = train_df.dropna()
    test_df = test_df.dropna()

    train_df = train_df[train_df["text"].str.len() > 0]
    test_df = test_df[test_df["text"].str.len() > 0]

    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

    print(f"Training samples: {len(train_df)}")
    print(f"Testing samples: {len(test_df)}")
    print("Created data/processed/train.csv")
    print("Created data/processed/test.csv")

if __name__ == "__main__":
    main()