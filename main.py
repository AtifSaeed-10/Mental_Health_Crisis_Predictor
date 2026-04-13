"""
Mental Health Crisis Predictor – FastAPI backend
Deploy this on Railway.  The Streamlit app sends these 14 raw features.
We compute the engineered features here before calling the model.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import warnings
import numpy as np
import os

warnings.filterwarnings("ignore")

app = FastAPI(title="Mental Health Crisis Predictor API")

model  = None
scaler = None


@app.on_event("startup")
async def load_model():
    global model, scaler
    try:
        model  = joblib.load("mental_health_model.pkl")
        scaler = joblib.load("scaler.pkl")
        print("✅  Model loaded successfully")
    except Exception as e:
        print(f"❌  Error loading model: {e}")
        model  = None
        scaler = None


# ── REQUEST SCHEMA ────────────────────────────────────────────────────────
# These are exactly the 14 raw columns the Streamlit form sends.
class PredictRequest(BaseModel):
    Gender:                  float   # Female=0, Male=1
    Occupation:              float   # Corporate=0, Self-Employed=1, Student=2, Other=3
    self_employed:           float   # No=0, Yes=1
    family_history:          float   # No=0, Yes=1
    Days_Indoors:            float   # 0-4 ordinal
    Growing_Stress:          float   # No=0, Maybe=0.5, Yes=1
    Changes_Habits:          float   # No=0, Maybe=0.5, Yes=1
    Mental_Health_History:   float   # No=0, Maybe=0.5, Yes=1
    Mood_Swings:             float   # Low=0, Medium=1, High=2
    Coping_Struggles:        float   # No=0, Yes=1
    Work_Interest:           float   # No=0, Maybe=0.5, Yes=1
    Social_Weakness:         float   # No=0, Maybe=0.5, Yes=1
    mental_health_interview: float   # No=0, Maybe=0.5, Yes=1
    care_options:            float   # No=0, Not sure=0.5, Yes=1


# ── HEALTH CHECK ──────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Mental Health API is running ✅"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "port": os.environ.get("PORT", "8000"),
    }


# ── PREDICT ───────────────────────────────────────────────────────────────
@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        return {"error": "Model not loaded. Check Railway logs."}

    try:
        r = req  # short alias

        # ── Engineered features (must match training notebook) ──
        stress_score     = r.Growing_Stress + r.Mood_Swings + r.Coping_Struggles
        behavioral_score = (r.Changes_Habits + r.Work_Interest +
                            r.Social_Weakness + r.Days_Indoors)
        awareness_score  = (r.Mental_Health_History +
                            r.mental_health_interview +
                            r.care_options)

        stress_x_family    = stress_score    * r.family_history
        care_x_family      = r.care_options  * r.family_history
        awareness_x_family = awareness_score * r.family_history
        gender_x_stress    = r.Gender        * stress_score

        high_risk_flag = int(
            stress_score     >= 3 and
            behavioral_score >= 4 and
            r.family_history == 1
        )

        # ── Feature vector (22 columns – same order as X in training) ──
        features = np.array([[
            r.Gender,
            r.Occupation,
            r.self_employed,
            r.family_history,
            r.Days_Indoors,
            r.Growing_Stress,
            r.Changes_Habits,
            r.Mental_Health_History,
            r.Mood_Swings,
            r.Coping_Struggles,
            r.Work_Interest,
            r.Social_Weakness,
            r.mental_health_interview,
            r.care_options,
            # engineered
            stress_score,
            behavioral_score,
            awareness_score,
            stress_x_family,
            care_x_family,
            awareness_x_family,
            gender_x_stress,
            high_risk_flag,
        ]])

        prediction  = model.predict(features)[0]
        proba       = model.predict_proba(features)[0]
        confidence  = float(max(proba))

        label       = "Yes" if int(prediction) == 1 else "No"
        explanation = (
            "Based on your answers, seeking professional mental health support is recommended."
            if int(prediction) == 1
            else "Your responses suggest a lower risk profile right now. Stay mindful!"
        )

        return {
            "prediction":  label,
            "confidence":  confidence,
            "explanation": explanation,
        }

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


# ── LOCAL RUN ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
