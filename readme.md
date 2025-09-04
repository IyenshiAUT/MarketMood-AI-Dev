# 🤖 MarketMood AI - Financial News Analyzer & Summarizer: AI/ML Repository

This repository contains the **AI model** and **training pipelines** for the **Financial News Analyzer &  Summarizer** project.  
It serves as the **single source of truth** for model development, experimentation, and versioning.

---

## 📁 Repository Structure
```
marketmood-ai-dev/
├── .github/
│ └── workflows/
│ └── train_and_promote.yml # CI/CD workflow
├── model_training/
│ ├── scripts/
│ │ ├── train_sentiment.py # FinBERT-based sentiment training
│ │ └── train_summarization.py # BART-based summarization training
│ └── requirements.txt # Training dependencies
├── mlops/
│ └── promote_models.py # MLflow model promotion logic
└── README.md # Project documentation
```


---

## 🎯 Purpose

The goal of this repository is to **train, evaluate, and manage NLP models** for financial news.  
We currently focus on two key models:

- **Sentiment Analysis** → Based on **FinBERT**, fine-tuned for financial news sentiment.
- **Summarization** → Based on **BART**, fine-tuned for financial news summarization.

Models that meet defined performance criteria are automatically **logged and promoted** to the **MLflow Model Registry** via CI/CD.  
Promoted models are then ready to be consumed by the **Application API**.

---

## 🚀 CI/CD Pipeline

The GitHub Actions workflow `.github/workflows/train_and_promote.yml` automates:

1. **Triggers**
   - Push to the `main` branch
   - Manual `workflow_dispatch`

2. **Setup**
   - Installs dependencies from `model_training/requirements.txt`

3. **Training**
   - Runs training scripts:
     ```bash
     python model_training/scripts/train_sentiment.py
     python model_training/scripts/train_summarization.py
     ```
   - Logs models and metrics to **MLflow**

4. **Promotion**
   - Executes `mlops/promote_models.py`
   - Promotes best-performing models to the **Production** stage in MLflow

---

## ▶️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/IyenshiAUT/MarketMood-AI-Dev.git
cd marketmood-ai-dev
### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
source venv/bin/activate # On Linux/MacOS
venv\Scripts\activate # On Windows
```
### 3. Install Dependencies
```bash
pip install -r app/requirements.txt
```
### 4. Configure MLflow server
Set the MLflow tracking URI and credentials in your environment variables:
```bash
export MLFLOW_TRACKING_URI=http://mlflow-server:5000
export MLFLOW_TRACKING_USERNAME=MLFLOW_TRACKING_USERNAME
export MLFLOW_TRACKING_PASSWORD=MLFLOW_TRACKING_PASSWORD
export MLFLOW_EXPERIMENT_NAME="financial-news-analyzer"
```
### 5. Run training scripts
```bash
python model_training/scripts/train_sentiment.py
python model_training/scripts/train_summarization.py
```
---
## 📈 Model Registry
We use the MLflow Model Registry as a central hub for managing model lifecycle:
- Track versions: Each trained model is versioned
- Transition stages: e.g., Staging → Production
- Reproducibility: Ensures consistency across deployments
- Serving compatibility: Production models are directly consumed by the downstream application

Once a model is promoted to Production, it becomes available for deployment.

## 🛠️ Tech Stack
- Python 3.11+
- PyTorch / Hugging Face Transformers
- MLflow
- GitHub Actions (CI/CD)