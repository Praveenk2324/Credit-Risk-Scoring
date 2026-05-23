# 💳 End-to-End MLOps Credit Risk Engine

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-Fargate_ECS-232F3E?logo=amazon-aws&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white)

A production-grade, cloud-native machine learning system designed to predict credit default risk. This project bridges the gap between a standard Jupyter Notebook and a true enterprise deployment by implementing a complete **Continuous Integration, Continuous Training, and Continuous Deployment (CI/CT/CD)** pipeline.

## 🏗️ System Architecture

This project is built across five distinct engineering pillars:

1. **The Machine Learning Core:** A dynamic PyTorch neural network (`TabularNet`) optimized via **Optuna** for automated hyperparameter tuning. It handles 12 financial features and dynamically scales human-readable data using a serialized `StandardScaler`.
2. **MLOps & Governance:** **DVC** (Data Version Control) tracks dataset lineage, while **MLflow** handles experiment tracking and model registry. An automated quality gate ensures only models achieving a Validation AUC > `0.80` are promoted to production.
3. **The Application Layer:** A high-performance **FastAPI** backend containerized with **Docker**, decoupled from a user-friendly **Streamlit** frontend dashboard.
4. **Automation (CI/CT):** A custom Continuous Training orchestrator (`trigger_ct.py`) automatically ingests new data batches, triggers DVC pipelines, retraining, and MLflow validation to combat **Data Drift**.
5. **Cloud Deployment (CD):** Automated **GitHub Actions** workflows build and push the Docker image to Docker Hub upon code commits. The API is hosted serverless on **AWS ECS (Fargate)**, while the UI is hosted on **Streamlit Community Cloud**.

---

## 📂 Repository Structure

```text
├── .github/workflows/
│   └── deploy.yml          # CI/CD Pipeline for Docker Hub
├── data/
│   ├── simulation_stream/  # Incoming batch data simulating real-world drifts
│   └── processed/          # DVC-tracked preprocessed datasets
├── src/
│   ├── app.py              # Streamlit Frontend Dashboard
│   ├── serve.py            # FastAPI Backend Application
│   ├── train.py            # PyTorch Model Training script
│   ├── tune.py             # Optuna Hyperparameter Optimization
│   ├── preprocess.py       # Data Cleaning, Imputation, and Scaling
│   ├── register.py         # MLflow Model Registry & Quality Gate
│   ├── export.py           # Extracts production weights & scaler for Docker
│   └── trigger_ct.py       # Continuous Training Orchestrator
├── Dockerfile              # Containerization instructions
├── dvc.yaml                # DVC Pipeline definition
├── params.yaml             # Hyperparameters for training
└── requirements.txt        # Python dependencies
```
## 🚀 How to Run Locally

### 1. Prerequisites
* Python 3.11+
* Docker Desktop
* Git

### 2. Setup & Installation
Clone the repository and install the dependencies:
```bash
git clone [https://github.com/praveenk2324/credit-risk-scoring.git](https://github.com/praveenk2324/credit-risk-scoring.git)
cd credit-risk-scoring
pip install -r requirements.txt
```

### 3. Simulate Continuous Training (CT)
Run the automated orchestrator to ingest data, trigger the DVC pipeline, train the model, validate the AUC, and extract the artifacts:

```bash
# Train on baseline data
python src/trigger_ct.py 01_base_data.csv

# Simulate an economic crash (Data Drift)
python src/trigger_ct.py 02_batch_2_drifted.csv
```
Note: If the drifted data causes the model to perform below the 0.80 AUC threshold, the pipeline will intentionally fail, preventing a degraded model from reaching production.

### 4. Run the Application
Start the FastAPI Backend (via Docker):

```bash
docker build -t credit-risk-api .
docker run -d -p 8000:8000 credit-risk-api
```
Start the Streamlit Frontend:

```bash
streamlit run src/app.py
```

### ☁️ Cloud Deployment Setup
This project uses a highly cost-optimized, decoupled cloud architecture.

* Backend: **AWS ECS (Fargate)**
The GitHub Action (deploy.yml) automatically builds and pushes the Docker image to Docker Hub on every push to main.

An AWS ECS Cluster pulls the :latest image and boots it on a serverless Fargate task (0.25 vCPU, 0.5 GB RAM).

The task exposes Port 8000 and assigns a Public IP address.

Cost Strategy: Desired tasks can be scaled to 0 to instantly pause billing when not in use.

* Frontend: **Streamlit Community Cloud**
Connected directly to this GitHub repository.

Automatically rebuilds the UI on code pushes.

Points to the AWS Public IP via the API_URL variable in app.py.