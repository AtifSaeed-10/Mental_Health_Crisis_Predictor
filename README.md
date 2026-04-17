# 🧠 Mental Health Crisis Predictor

A production-style machine learning system that predicts whether a person is likely to seek mental health treatment based on behavioral, emotional, and workplace-related factors.

🔗 **[Try the Live UI (Streamlit)](https://kgzcuuxfdexrmku9duunoq.streamlit.app/)**
---
##  System Architecture

![Architecture Diagram](docs/architecture.png)
## 🚀 System Overview

This project has been upgraded from a simple ML app to a **full-stack AI system**:

* **Frontend (Streamlit)** → User interface for input & visualization
* **Backend (FastAPI)** → Handles requests and serves predictions
* **Deployment (Railway)** → Hosts the API for public access

### 🔁 Architecture

User → Streamlit UI → FastAPI API → ML Model → JSON Response

---

## 📌 What It Does

Users fill out a short form covering:

* stress levels
* mood changes
* behavioral patterns
* work environment
* mental health history

The system returns a prediction indicating whether **mental health treatment is likely recommended**.

> ⚠️ This is not a medical diagnosis — it is a pattern-based ML prediction designed for awareness.

---

## ⚙️ How It Works

### 1. Data Preprocessing

* Missing values handled (e.g., self-employment field)
* Duplicates removed
* Irrelevant columns dropped (timestamp, country)

### 2. Feature Engineering

Raw survey responses were transformed into structured signals:

* **Stress Score** → stress + mood swings + coping ability
* **Behavioral Score** → habits + work interest + social activity
* **Awareness Score** → history + openness + care access

A **high-risk flag** is triggered when all dimensions are elevated along with family history.

---

### 3. Model Training

Multiple models were evaluated:

* Logistic Regression
* Random Forest
* XGBoost
* KNN
* Gradient Boosting

---

### 4. Final Model

A **Voting Classifier (Soft Voting)** combining:

* Logistic Regression
* XGBoost
* Gradient Boosting

Selected based on best overall performance.

---

## 🧠 API (FastAPI Backend)

The model is exposed as a REST API.

### 🔹 Endpoint: `/predict`

* **Method:** POST
* **Input:**

```json
{
  "text": "I feel stressed and isolated"
}
```

* **Response:**

```json
{
  "prediction": "likely_needs_treatment"
}
```

---

### 🔹 Endpoint: `/health`

* **Method:** GET
* **Purpose:** Check if API is running

---

### 🛠 Tech Details

* FastAPI (ASGI framework)
* Uvicorn (server)
* Pydantic (input validation)
* JSON-based request/response

---

## 📂 Project Structure

| File                       | Description                                            |
| -------------------------- | ------------------------------------------------------ |
| `MentalHealthCrisis.ipynb` | Training pipeline (EDA + feature engineering + models) |
| `mental_health_model.pkl`  | Saved voting classifier                                |
| `scaler.pkl`               | Feature scaling object                                 |
| `app.py`                   | Streamlit frontend                                     |
| `main.py`                  | FastAPI backend                                        |

---

## 🧰 Tech Stack

* Python
* pandas, numpy
* scikit-learn
* XGBoost
* FastAPI
* Streamlit
* Uvicorn
* Railway
* joblib

---

## 🌍 Deployment

* **Frontend:** Streamlit Cloud
* **Backend:** Railway (FastAPI API)

---

## ⚠️ Disclaimer

This tool is for educational and awareness purposes only.
It does not provide medical advice, diagnosis, or treatment.

---

## 💡 Key Insight

> This project demonstrates how a machine learning model can be transformed into a **deployable, API-driven product** rather than just a local experiment.
