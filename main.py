from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import warnings
import numpy as np

warnings.filterwarnings('ignore')

app = FastAPI(title="Mental Health Crisis Predictor API")

# Load model
try:
    model = joblib.load("mental_health_model.pkl")
    scaler = joblib.load("scaler.pkl")
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")

class PredictRequest(BaseModel):
    stress_level: float
    mood_swings: float
    coping_struggles: float
    habit_changes: float
    work_interest: float
    social_withdrawal: float
    days_indoors: float
    mental_health_history: float
    interview_comfort: float
    care_access: float
    family_history: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "Mental Health Crisis Predictor API is running"}

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        # Feature engineering - same as training notebook
        stress_score = (request.stress_level + request.mood_swings + request.coping_struggles) / 3
        behavioral_score = (request.habit_changes + (1 - request.work_interest) + request.social_withdrawal + (request.days_indoors / 30)) / 4
        awareness_score = (request.mental_health_history + request.interview_comfort + (1 - request.care_access)) / 3
        high_risk_flag = 1 if (stress_score > 0.6 and behavioral_score > 0.6 and awareness_score > 0.6 and request.family_history == 1) else 0
        
        # Create 22 features
        features = np.array([
            request.stress_level,
            request.mood_swings,
            request.coping_struggles,
            request.habit_changes,
            request.work_interest,
            request.social_withdrawal,
            request.days_indoors,
            request.mental_health_history,
            request.interview_comfort,
            request.care_access,
            request.family_history,
            stress_score,
            behavioral_score,
            awareness_score,
            high_risk_flag,
            request.stress_level * request.family_history,
            request.mood_swings * request.family_history,
            request.social_withdrawal * request.family_history,
            request.habit_changes * request.work_interest,
            request.days_indoors * request.social_withdrawal,
            request.mental_health_history * request.care_access,
            stress_score * request.family_history
        ]).reshape(1, -1)
        
        prediction = model.predict(features)[0]
        confidence = max(model.predict_proba(features)[0])

        return {
            "prediction": "Yes" if prediction == 1 else "No",
            "confidence": float(confidence),
            "explanation": "Mental health treatment likely needed" if prediction == 1 else "Low risk"
        }
    except Exception as e:
        return {"error": str(e), "type": type(e).__name__}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
