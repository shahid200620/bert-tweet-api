# BERT Tweet Sentiment API

A practical NLP project that turns text into a simple **positive or negative sentiment prediction** using a fine-tuned DistilBERT model.

Built as part of a **PATNR / GPP Data Science project**, this project goes beyond model training by connecting the trained NLP model to a FastAPI backend and a Streamlit web application.

## 🌐 Live Demo

🚀 **Try the live application:**
https://bert-tweet-sentiment.streamlit.app/

The live application allows you to enter text and instantly receive:

* Predicted sentiment
* Confidence score
* Simple visual feedback

> **Note:** The live demo is hosted on Streamlit and availability may depend on the deployment environment. The complete implementation is available in the GitHub repository.

## 📂 Source Code

**GitHub Repository:**
https://github.com/shahid200620/bert-tweet-api

---

## 🧠 Project Overview

Sentiment analysis is a practical application of Natural Language Processing (NLP) that helps machines understand whether a piece of text expresses a positive or negative opinion.

In this project, I built an end-to-end sentiment analysis system that:

* Preprocesses and prepares text data
* Fine-tunes a transformer-based DistilBERT model
* Evaluates the trained model using standard classification metrics
* Saves the trained model for inference
* Exposes the model through a FastAPI REST API
* Provides an interactive Streamlit web interface
* Supports Docker-based deployment

The goal was not simply to train a model, but to understand how an NLP model can move from an experiment into a usable application.

## ✨ What It Does

The application accepts text such as:

> "This product is absolutely amazing!"

and produces a prediction such as:

```json
{
  "sentiment": "positive",
  "confidence": 0.97
}
```

For negative text, the system can return:

```json
{
  "sentiment": "negative",
  "confidence": 0.91
}
```

The Streamlit interface presents the result in a simple and user-friendly format.

---

## 🤖 Model

The project uses:

**DistilBERT — `distilbert-base-uncased`**

DistilBERT is a smaller and faster version of BERT that retains much of BERT's language understanding capability while being more practical for applications with limited computational resources.

The pretrained model was fine-tuned specifically for **binary sentiment classification**.

### Training Configuration

| Parameter            |                           Value |
| -------------------- | ------------------------------: |
| Model                |       `distilbert-base-uncased` |
| Learning Rate        |                          `2e-5` |
| Batch Size           |                             `8` |
| Epochs               |                             `1` |
| Maximum Input Length |                            `96` |
| Task                 | Binary Sentiment Classification |

---

## 📊 Model Performance

The final model achieved the following evaluation results:

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **83.00%** |
| Precision | **86.18%** |
| Recall    | **78.60%** |
| F1 Score  | **82.22%** |

The results show that the model is able to identify the overall sentiment of unseen text with a good balance between precision and recall.

The evaluation results are also stored in:

```text
results/metrics.json
```

---

## 🏗️ Architecture

The project follows a simple end-to-end architecture:

```text
                    ┌──────────────────┐
                    │    User Text     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    Streamlit UI  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    FastAPI API   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    Tokenizer     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    DistilBERT    │
                    │  Sentiment Model │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Prediction +     │
                    │ Confidence Score │
                    └──────────────────┘
```

This separation between the **UI, API and ML model** keeps the project organized and makes it easier to maintain or extend.

---

## 🚀 Features

### 🔹 Transformer-Based NLP

Uses a fine-tuned DistilBERT model for sentiment classification rather than relying only on traditional machine learning techniques.

### 🔹 REST API

FastAPI provides an API layer through which applications can send text and receive sentiment predictions.

### 🔹 Interactive Web Interface

The Streamlit application allows users to enter text and see the prediction without needing to interact directly with the API.

### 🔹 Confidence Score

Along with the predicted sentiment, the system provides a confidence value representing how strongly the model favors its prediction.

### 🔹 Saved Model

The trained model and tokenizer are stored in `model_output/`, allowing the application to load the trained model directly without retraining.

### 🔹 Evaluation Results

Metrics and prediction outputs are stored under `results/` for reference and reproducibility.

### 🔹 Docker Support

Dockerfiles and Docker Compose configuration are included for running the application in a containerized environment.

---

## 📁 Project Structure

```text
bert-tweet-api/
│
├── data/
│   ├── processed/
│   └── unseen/
│
├── model_output/
│   ├── config.json
│   ├── model.safetensors
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── training_args.bin
│   └── vocab.txt
│
├── results/
│   ├── metrics.json
│   ├── predictions.csv
│   ├── run_summary.json
│   └── test_predictions.csv
│
├── scripts/
│   ├── batch_predict.py
│   ├── preprocess.py
│   └── train.py
│
├── src/
│   ├── api.py
│   ├── ui.py
│   └── ui_cloud.py
│
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile.api
├── Dockerfile.ui
├── docker-compose.yml
├── README.md
└── requirements.txt
```

---

## 🔄 Project Workflow

### 1. Data Preparation

The text data is prepared and processed before being passed to the transformer model.

The preprocessing stage ensures that the dataset is in a suitable format for training and evaluation.

### 2. Model Training

The processed dataset is used to fine-tune `distilbert-base-uncased` for binary sentiment classification.

The training pipeline is available in:

```text
scripts/train.py
```

### 3. Model Evaluation

After training, predictions are generated on evaluation data and standard classification metrics are calculated.

The results are saved in:

```text
results/metrics.json
```

### 4. Model Saving

The fine-tuned model and tokenizer are stored in:

```text
model_output/
```

This allows the trained model to be loaded directly during inference.

### 5. API Development

FastAPI provides endpoints for checking the service and performing sentiment predictions.

#### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "ok"
}
```

#### Sentiment Prediction

```http
POST /predict
```

Example request:

```json
{
  "text": "I really enjoyed this experience!"
}
```

Example response:

```json
{
  "sentiment": "positive",
  "confidence": 0.95
}
```

### 6. Streamlit Interface

The Streamlit application provides a simple interface where users can enter text and analyze its sentiment.

The UI communicates with the FastAPI backend, keeping the presentation layer separate from the model-serving layer.

---

## 🛠️ Tech Stack

### Machine Learning & NLP

* Python
* PyTorch
* Hugging Face Transformers
* DistilBERT
* Scikit-learn
* Pandas

### Backend

* FastAPI
* Pydantic
* Uvicorn

### Frontend

* Streamlit

### Deployment

* Docker
* Docker Compose
* Streamlit Community Cloud

### Development

* Git
* GitHub
* Python virtual environment
* Environment variables

---

## 💻 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/shahid200620/bert-tweet-api.git
cd bert-tweet-api
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file using:

```text
.env.example
```

The environment configuration allows the API and Streamlit application to locate the saved model and communicate with the correct backend URL.

### 5. Start the FastAPI server

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

The API will be available at:

```text
http://127.0.0.1:8000
```

FastAPI's interactive documentation will be available at:

```text
http://127.0.0.1:8000/docs
```

### 6. Start Streamlit

Open another terminal and run:

```bash
streamlit run src/ui.py
```

The Streamlit application will then open in your browser.

---

## 🐳 Docker

The repository includes Docker configuration for both the API and UI.

To build and start the application:

```bash
docker compose up --build
```

Docker provides a consistent environment for running the different components of the project.

---

## 🔌 API Example

The prediction endpoint can also be used programmatically.

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"text": "The experience was really good!"}
)

print(response.json())
```

Example:

```json
{
  "sentiment": "positive",
  "confidence": 0.94
}
```

---

## 🎯 Why I Built This Project

The main objective was to understand the complete lifecycle of a practical NLP application.

Instead of stopping after training a model and checking its accuracy, I wanted to take the next steps:

**Data → Preprocessing → Fine-tuning → Evaluation → Model Saving → API → UI → Deployment**

This helped me understand how machine learning models can be converted into applications that real users can actually interact with.

---

## 📚 Key Learning Outcomes

Through this project, I gained practical experience with:

* Natural Language Processing
* Transformer architectures
* DistilBERT
* Fine-tuning pretrained models
* Text classification
* Model evaluation
* PyTorch inference
* Hugging Face Transformers
* REST API development
* FastAPI
* Streamlit
* Docker
* Model deployment
* Environment variable management
* Separating ML logic from application logic

One of the biggest takeaways was that **building an ML project is more than achieving a good accuracy score**. Making the model usable, accessible and deployable is an equally important part of the process.

---

## 🔮 Future Improvements

Possible improvements for a future version include:

* Supporting positive, neutral and negative sentiment
* Training on a larger and more diverse dataset
* Adding batch prediction through the API
* Adding automated API testing
* Adding CI/CD with GitHub Actions
* Improving logging and monitoring
* Adding API authentication and rate limiting
* Deploying the backend and frontend using a dedicated cloud architecture
* Improving model performance through further fine-tuning

---

## 📌 Project Links

| Resource             | Link                                           |
| -------------------- | ---------------------------------------------- |
| 🚀 Live Demo         | https://bert-tweet-sentiment.streamlit.app/    |
| 💻 GitHub Repository | https://github.com/shahid200620/bert-tweet-api |

---

## 👨‍💻 Author

**Shahid Mohammed**

Computer Science Engineering Student
India

**GitHub:**
https://github.com/shahid200620

---

## ❤️ Final Note

This project started as a sentiment analysis task, but it became an opportunity to understand the complete journey of a machine learning application.

From preparing the data and fine-tuning a transformer model to building an API, creating a user interface and deploying the final application, every stage helped bridge the gap between **machine learning experimentation and real-world application development**.

