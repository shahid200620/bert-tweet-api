import os
import json
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

MODEL_NAME = "distilbert-base-uncased"
MODEL_PATH = "model_output"
RESULTS_PATH = "results"
MAX_LENGTH = 96
LEARNING_RATE = 2e-5
BATCH_SIZE = 8
EPOCHS = 1

def load_data():
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    train_df = train_df[["text", "label"]].dropna()
    test_df = test_df[["text", "label"]].dropna()

    train_df = train_df.groupby("label", group_keys=False).sample(n=2000, random_state=42)
    test_df = test_df.groupby("label", group_keys=False).sample(n=500, random_state=42)

    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return Dataset.from_pandas(train_df, preserve_index=False), Dataset.from_pandas(test_df, preserve_index=False)

def tokenize_data(train_data, test_data, tokenizer):
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

    train_data = train_data.map(tokenize, batched=True)
    test_data = test_data.map(tokenize, batched=True)

    train_data = train_data.remove_columns(["text"])
    test_data = test_data.remove_columns(["text"])

    train_data.set_format("torch")
    test_data.set_format("torch")

    return train_data, test_data

def compute_metrics(pred):
    predictions = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary", zero_division=0)
    accuracy = accuracy_score(labels, predictions)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }

def main():
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(RESULTS_PATH, exist_ok=True)

    train_data, test_data = load_data()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    train_data, test_data = tokenize_data(train_data, test_data, tokenizer)

    training_args = TrainingArguments(
        output_dir="results/training",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    evaluation = trainer.evaluate()
    metrics = {
        "accuracy": float(evaluation["eval_accuracy"]),
        "precision": float(evaluation["eval_precision"]),
        "recall": float(evaluation["eval_recall"]),
        "f1_score": float(evaluation["eval_f1_score"])
    }

    trainer.save_model(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)

    with open("results/metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    run_summary = {
        "hyperparameters": {
            "model_name": MODEL_NAME,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "num_epochs": EPOCHS
        },
        "final_metrics": {
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"]
        }
    }

    with open("results/run_summary.json", "w") as file:
        json.dump(run_summary, file, indent=4)

    print("Training completed")
    print("Accuracy:", metrics["accuracy"])
    print("Precision:", metrics["precision"])
    print("Recall:", metrics["recall"])
    print("F1 Score:", metrics["f1_score"])
    print("Model saved to:", MODEL_PATH)

if __name__ == "__main__":
    main()