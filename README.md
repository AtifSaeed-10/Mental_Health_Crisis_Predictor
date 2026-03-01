# 🧠 Mental Health Crisis Predictor

A machine learning project that predicts whether a person is likely to seek mental health treatment based on personal, behavioral, and workplace-related factors.

🔗 **[Try the Live App](https://kgzcuuxfdexrmku9duunoq.streamlit.app/)**

---

## What It Does

You fill out a short form covering things like stress levels, mood changes, work habits, and family history — and the model tells you whether treatment is likely recommended or not.

It's not a diagnosis. It's a pattern-based prediction meant to raise awareness and encourage people to take their mental health seriously.

---

## How It Works

The model was trained on a mental health survey dataset. Here's a quick breakdown of the pipeline:

1. **Data Cleaning** — Handled missing values (self-employment field filled by mode), removed duplicates, dropped irrelevant columns like timestamp and country.

2. **Feature Engineering** — Raw survey answers were mapped to numbers, then grouped into three composite scores:
   - **Stress Score** — growing stress + mood swings + coping struggles
   - **Behavioral Score** — habit changes + work interest + social withdrawal + days indoors
   - **Awareness Score** — mental health history + interview comfort + care access

   A **high-risk flag** is also computed when all three areas are elevated alongside family history.

3. **Models Trained** — Five classifiers were compared:
   - Logistic Regression
   - Random Forest
   - XGBoost
   - KNN
   - Gradient Boosting

4. **Final Model** — A **Voting Classifier** (soft voting) combining Logistic Regression, XGBoost, and Gradient Boosting. This gave the best overall accuracy.

---

## Files

| File | Description |
|------|-------------|
| `MentalHealthCrisis.ipynb` | Full training notebook with EDA, feature engineering, and model comparison |
| `mental_health_model.pkl` | Saved voting classifier |
| `scaler.pkl` | StandardScaler used for Logistic Regression inputs |
| `app.py` | Streamlit frontend for the live prediction app |

---

## Tech Stack

- Python
- pandas, numpy
- scikit-learn
- XGBoost
- Streamlit
- joblib

---


## Dataset

The model was trained on a publicly available mental health survey dataset covering demographic info, workplace environment, stress indicators, and treatment history.

---

## Disclaimer

This tool is for informational and educational purposes only. It is not a substitute for professional mental health advice, diagnosis, or treatment.
