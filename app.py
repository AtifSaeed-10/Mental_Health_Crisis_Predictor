import streamlit as st
import requests
import json

st.set_page_config(
    page_title="Mental Health Crisis Predictor",
    page_icon="🧠",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@200;300;400;600;700;800&family=Space+Mono:wght@400;700&display=swap');

:root {
    --ink:      #0b0d17;
    --surface:  #111527;
    --card:     #161b2e;
    --border:   #1f2a45;
    --accent1:  #7f5af0;
    --accent2:  #2cb67d;
    --accent3:  #ff8906;
    --text:     #fffffe;
    --muted:    #72757e;
}

* { font-family: 'Outfit', sans-serif; box-sizing: border-box; }

.stApp {
    background-color: var(--ink);
    background-image:
        radial-gradient(ellipse 80% 60% at 20% 10%,  rgba(127,90,240,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 80%,  rgba(44,182,125,0.14) 0%, transparent 55%),
        radial-gradient(ellipse 50% 40% at 60% 30%,  rgba(255,137,6,0.08)  0%, transparent 50%),
        url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.015'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    color: var(--text);
    min-height: 100vh;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 4rem; position: relative; z-index: 1; }

.hero { text-align: center; padding: 2.5rem 1rem 1.5rem; }
.hero-brain { font-size: 5rem; display: block; margin-bottom: 0.8rem; filter: drop-shadow(0 0 30px rgba(127,90,240,0.6)); }
.hero h1 { font-size: 3rem; font-weight: 800; line-height: 1.1; margin: 0 0 0.5rem; background: linear-gradient(135deg, #c4b5fd 0%, #fffffe 45%, #6ee7b7 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero-sub { color: var(--muted); font-size: 1rem; font-weight: 300; margin-bottom: 0.5rem; }

.result-box {
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-top: 1rem;
    animation: resultReveal 0.6s cubic-bezier(0.34,1.56,0.64,1) both;
}

.result-treatment {
    background: linear-gradient(135deg, #1a0f3a 0%, #0f1628 100%);
    border: 1px solid rgba(127,90,240,0.5);
    box-shadow: 0 0 60px rgba(127,90,240,0.15);
}
.result-ok {
    background: linear-gradient(135deg, #0a2419 0%, #0f1628 100%);
    border: 1px solid rgba(44,182,125,0.5);
    box-shadow: 0 0 60px rgba(44,182,125,0.12);
}

@keyframes resultReveal {
    from { opacity: 0; transform: scale(0.88) translateY(20px); }
    to   { opacity: 1; transform: scale(1) translateY(0); }
}

.result-icon  { font-size: 4.5rem; display: block; margin-bottom: 0.6rem; }
.result-title { font-size: 1.8rem; font-weight: 800; margin-bottom: 0.4rem; }
.result-treatment .result-title { color: #c4b5fd; }
.result-ok .result-title        { color: #6ee7b7; }

.result-desc {
    color: var(--muted);
    font-size: 0.9rem;
    font-weight: 300;
    max-width: 340px;
    margin: 0 auto 1.2rem;
    line-height: 1.6;
}

.conf-wrap {
    background: rgba(0,0,0,0.3);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
}
.conf-pct {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
}
.result-treatment .conf-pct { color: #a78bfa; }
.result-ok .conf-pct        { color: #2cb67d; }

.conf-track {
    background: rgba(255,255,255,0.06);
    border-radius: 999px;
    height: 6px;
    overflow: hidden;
}
.conf-fill-t {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #7f5af0, #c4b5fd);
    box-shadow: 0 0 8px rgba(127,90,240,0.6);
}
.conf-fill-ok {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #2cb67d, #6ee7b7);
    box-shadow: 0 0 8px rgba(44,182,125,0.6);
}

.disclaimer {
    background: rgba(255,137,6,0.06);
    border: 1px solid rgba(255,137,6,0.2);
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    margin-top: 1.2rem;
    font-size: 0.78rem;
    color: #a8956a;
    line-height: 1.5;
}

div[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, #7f5af0 0%, #2cb67d 100%);
    color: white;
    font-family: 'Outfit', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    padding: 0.9rem 2rem;
    border: none;
    border-radius: 14px;
    cursor: pointer;
    margin-top: 1.5rem;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(127,90,240,0.45);
}
</style>
""", unsafe_allow_html=True)

# ─── HERO ────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <span class="hero-brain">🧠</span>
    <h1>Mental Health<br>Crisis Predictor</h1>
    <p class="hero-sub">Powered by Railway API</p>
</div>
""", unsafe_allow_html=True)

# ─── FORM ────────────────────────────────────────────────────
st.markdown("### 📋 Assessment Form")

col1, col2 = st.columns(2)
with col1:
    stress_level = st.slider("Stress Level (0-1)", 0.0, 1.0, 0.5, 0.1)
    mood_swings = st.slider("Mood Swings (0-1)", 0.0, 1.0, 0.5, 0.1)
    coping_struggles = st.slider("Coping Struggles (0-1)", 0.0, 1.0, 0.5, 0.1)

with col2:
    habit_changes = st.slider("Habit Changes (0-1)", 0.0, 1.0, 0.5, 0.1)
    work_interest = st.slider("Work Interest (0-1)", 0.0, 1.0, 0.5, 0.1)
    social_withdrawal = st.slider("Social Withdrawal (0-1)", 0.0, 1.0, 0.5, 0.1)

col3, col4 = st.columns(2)
with col3:
    days_indoors = st.slider("Days Indoors (0-30)", 0, 30, 15)
    mental_health_history = st.selectbox("Mental Health History?", [0, 1])

with col4:
    interview_comfort = st.slider("Interview Comfort (0-1)", 0.0, 1.0, 0.5, 0.1)
    care_access = st.slider("Care Access (0-1)", 0.0, 1.0, 0.5, 0.1)

family_history = st.selectbox("Family History?", [0, 1])

# ─── PREDICT ─────────────────────────────────────────────────
predict_btn = st.button("⚡ Analyze My Mental Health Risk")

if predict_btn:
    with st.spinner("Connecting to Railway API... 🚀"):
        try:
            API_URL = "https://mentalhealthcrisispredictor-production.up.railway.app/predict"
            
            payload = {
                "stress_level": float(stress_level),
                "mood_swings": float(mood_swings),
                "coping_struggles": float(coping_struggles),
                "habit_changes": float(habit_changes),
                "work_interest": float(work_interest),
                "social_withdrawal": float(social_withdrawal),
                "days_indoors": float(days_indoors),
                "mental_health_history": float(mental_health_history),
                "interview_comfort": float(interview_comfort),
                "care_access": float(care_access),
                "family_history": float(family_history)
            }
            
            response = requests.post(API_URL, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                confidence = result["confidence"] * 100
                prediction = result["prediction"]
                
                if prediction == "Yes":
                    st.markdown(f"""
                    <div class="result-box result-treatment">
                        <span class="result-icon">🔴</span>
                        <div class="result-title">Treatment Recommended</div>
                        <p class="result-desc">Professional mental health support could be beneficial.</p>
                        <div class="conf-wrap">
                            <div class="conf-pct">{confidence:.1f}% Confidence</div>
                            <div class="conf-track">
                                <div class="conf-fill-t" style="width:{confidence}%"></div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-box result-ok">
                        <span class="result-icon">🟢</span>
                        <div class="result-title">No Treatment Needed</div>
                        <p class="result-desc">No immediate intervention appears necessary.</p>
                        <div class="conf-wrap">
                            <div class="conf-pct">{confidence:.1f}% Confidence</div>
                            <div class="conf-track">
                                <div class="conf-fill-ok" style="width:{confidence}%"></div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="disclaimer">
                    ⚠️ <strong>Disclaimer:</strong> For educational purposes only. Consult a professional.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"API Error: {response.status_code}")
        
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timeout. Railway might be sleeping. Try again!")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
