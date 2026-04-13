from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import warnings
import numpy as np
import traceback
import sys

warnings.filterwarnings('ignore')

app = FastAPI(title="Mental Health Crisis Predictor API")

model = None
scaler = None

print("Script started", file=sys.stderr)

# Load model at startup
@app.on_event("startup")
async def load_model():
    global model, scaler
    try:
        print("Loading model...", file=sys.stderr)
        model = joblib.load("mental_health_model.pkl")
        scaler = joblib.load("scaler.pkl")
        print("✅ Model loaded successfully", file=sys.stderr)
    except Exception as e:
        print(f"❌ Error loading model: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        model = None
        scaler = None

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
    try:
        print("Health check called", file=sys.stderr)
        return {"status": "ok", "model_loaded": model is not None}
    except Exception as e:
        print(f"Error in health: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return {"status": "error", "error": str(e)}

@app.get("/")
def root():
    return {"message": "Mental Health Crisis Predictor API is running"}

@app.post("/predict")
def predict(request: PredictRequest):
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        stress_score = (request.stress_level + request.mood_swings + request.coping_struggles) / 3
        behavioral_score = (request.habit_changes + (1 - request.work_interest) + request.social_withdrawal + (request.days_indoors / 30)) / 4
        awareness_score = (request.mental_health_history + request.interview_comfort + (1 - request.care_access)) / 3
        high_risk_flag = 1 if (stress_score > 0.6 and behavioral_score > 0.6 and awareness_score > 0.6 and request.family_history == 1) else 0
        
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
        traceback.print_exc(file=sys.stderr)
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
