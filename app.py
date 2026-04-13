import os
import streamlit as st
import requests

st.set_page_config(
    page_title="Mental Health Crisis Predictor",
    page_icon="🧠",
    layout="centered"
)

API_URL = (
    st.secrets.get("API_URL", None)
    if hasattr(st, "secrets")
    else None
) or os.getenv("API_URL") or "https://mentalhealthcrisispredictor-production.up.railway.app/predict"

REQUEST_TIMEOUT = 20

# ── STYLES ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg:      #0c0e1a;
    --card:    #13162a;
    --border:  rgba(255,255,255,0.07);
    --accent:  #6ee7f7;
    --warn:    #ff6b6b;
    --ok:      #57e0a0;
    --muted:   #7a80a0;
    --text:    #e8ecf7;
}

* { font-family: 'Syne', sans-serif; box-sizing: border-box; }
.stApp { background: var(--bg); color: var(--text); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem; padding-bottom: 3rem; max-width: 820px; }

/* ── HERO ── */
.hero { text-align: center; padding: 2rem 0 1.5rem; }
.hero-icon { font-size: 3.8rem; display: block; margin-bottom: 0.5rem; }
.hero h1 {
    font-size: 2.4rem; font-weight: 800; margin: 0;
    background: linear-gradient(90deg, var(--accent) 0%, #a78bfa 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub { color: var(--muted); font-size: 0.9rem; margin-top: 0.5rem; font-family: 'DM Mono', monospace; }

/* ── SECTION LABEL ── */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem; letter-spacing: 0.15em; text-transform: uppercase;
    color: var(--accent); margin: 1.8rem 0 0.6rem;
    border-left: 3px solid var(--accent); padding-left: 0.6rem;
}

/* ── RESULT BOX ── */
.result-box {
    border-radius: 14px; padding: 1.4rem 1.2rem;
    margin-top: 1.2rem; text-align: center;
    animation: fadeIn 0.4s ease;
}
@keyframes fadeIn { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
.result-ok  { background: #081b12; border: 1px solid rgba(87,224,160,0.4); }
.result-bad { background: #1a0c1c; border: 1px solid rgba(255,107,107,0.4); }
.result-title { font-size: 1.4rem; font-weight: 800; margin-bottom: 0.3rem; }
.result-ok  .result-title { color: var(--ok); }
.result-bad .result-title { color: var(--warn); }
.result-conf { font-family: 'DM Mono', monospace; font-size: 0.85rem; color: var(--muted); }
.result-desc { color: #c2c9e0; font-size: 0.92rem; margin-top: 0.4rem; }

/* ── DISCLAIMER ── */
.disclaimer {
    background: rgba(255,180,0,0.06); border: 1px solid rgba(255,180,0,0.22);
    border-radius: 10px; padding: 0.8rem 1rem; margin-top: 1rem;
    font-size: 0.78rem; color: #c8a96e; line-height: 1.5;
}

/* ── BUTTON ── */
div[data-testid="stButton"] > button {
    width: 100% !important;
    background: linear-gradient(135deg, #6ee7f7 0%, #a78bfa 100%) !important;
    color: #0c0e1a !important; font-size: 1rem !important;
    font-weight: 800 !important; padding: 0.85rem 1.5rem !important;
    border: none !important; border-radius: 12px !important;
    margin-top: 1rem !important; letter-spacing: 0.03em !important;
    transition: opacity 0.2s !important;
}
div[data-testid="stButton"] > button:hover { opacity: 0.88 !important; }

/* ── SLIDER LABELS ── */
label[data-testid="stWidgetLabel"] p { color: var(--text) !important; font-size: 0.88rem !important; }

/* ── SELECT BOX ── */
div[data-baseweb="select"] { background: var(--card) !important; }
</style>
""", unsafe_allow_html=True)

# ── HERO ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <span class="hero-icon">🧠</span>
    <h1>Mental Health Crisis Predictor</h1>
    <p class="hero-sub">answer honestly · takes ~60 seconds · not a diagnosis</p>
</div>
""", unsafe_allow_html=True)

# ── FORM ────────────────────────────────────────────────────────────────────

# Helper maps
YESNO        = {"Yes": 1, "No": 0}
YESNO_MAYBE  = {"Yes": 1, "Maybe": 0.5, "No": 0}
GENDER_MAP   = {"Female": 0, "Male": 1}
OCC_MAP      = {"Corporate": 0, "Self-Employed": 1, "Student": 2, "Other": 3}
DAYS_MAP     = {"Go out every day": 0, "1-14 days": 1, "15-30 days": 2,
                "31-60 days": 3, "More than 2 months": 4}
MOOD_MAP     = {"Low": 0, "Medium": 1, "High": 2}
CARE_MAP     = {"No": 0, "Not sure": 0.5, "Yes": 1}

# ── Section 1: About You
st.markdown('<div class="section-label">01 · About You</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    gender     = st.selectbox("Gender", list(GENDER_MAP.keys()))
    occupation = st.selectbox("Occupation", list(OCC_MAP.keys()))
with col2:
    self_employed   = st.selectbox("Self-Employed?", list(YESNO.keys()))
    family_history  = st.selectbox("Family history of mental illness?", list(YESNO.keys()))

# ── Section 2: Stress & Mood
st.markdown('<div class="section-label">02 · Stress & Mood</div>', unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    growing_stress    = st.selectbox("Growing stress lately?", list(YESNO_MAYBE.keys()))
    mood_swings       = st.selectbox("Mood swing intensity?", list(MOOD_MAP.keys()))
with col4:
    coping_struggles  = st.selectbox("Struggling to cope?", list(YESNO.keys()))
    changes_habits    = st.selectbox("Changes in habits?", list(YESNO_MAYBE.keys()))

# ── Section 3: Behaviour
st.markdown('<div class="section-label">03 · Behaviour & Social Life</div>', unsafe_allow_html=True)
col5, col6 = st.columns(2)
with col5:
    work_interest     = st.selectbox("Lost interest in work?", list(YESNO_MAYBE.keys()))
    social_weakness   = st.selectbox("Feeling socially weak/withdrawn?", list(YESNO_MAYBE.keys()))
with col6:
    days_indoors      = st.selectbox("Days spent indoors (last month)?", list(DAYS_MAP.keys()))

# ── Section 4: Awareness
st.markdown('<div class="section-label">04 · Awareness & Support</div>', unsafe_allow_html=True)
col7, col8 = st.columns(2)
with col7:
    mh_history        = st.selectbox("Have you had mental health issues before?", list(YESNO_MAYBE.keys()))
    mh_interview      = st.selectbox("Comfortable discussing MH in job interviews?", list(YESNO_MAYBE.keys()))
with col8:
    care_options      = st.selectbox("Do you have access to care/treatment?", list(CARE_MAP.keys()))

# ── PREDICT BUTTON ──────────────────────────────────────────────────────────
if st.button("⚡  Predict Now"):
    # Build numeric feature dict (must match main.py exactly)
    payload = {
        "Gender":                  GENDER_MAP[gender],
        "Occupation":              OCC_MAP[occupation],
        "self_employed":           YESNO[self_employed],
        "family_history":          YESNO[family_history],
        "Days_Indoors":            DAYS_MAP[days_indoors],
        "Growing_Stress":          YESNO_MAYBE[growing_stress],
        "Changes_Habits":          YESNO_MAYBE[changes_habits],
        "Mental_Health_History":   YESNO_MAYBE[mh_history],
        "Mood_Swings":             MOOD_MAP[mood_swings],
        "Coping_Struggles":        YESNO[coping_struggles],
        "Work_Interest":           YESNO_MAYBE[work_interest],
        "Social_Weakness":         YESNO_MAYBE[social_weakness],
        "mental_health_interview": YESNO_MAYBE[mh_interview],
        "care_options":            CARE_MAP[care_options],
    }

    with st.spinner("Talking to Railway API... 🚀"):
        try:
            response = requests.post(API_URL, json=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()

            prediction  = data.get("prediction", "Unknown")
            confidence  = data.get("confidence", None)
            explanation = data.get("explanation", "")

            is_high_risk = str(prediction).strip().lower() in {"yes", "1", "true", "treatment recommended"}

            if is_high_risk:
                conf_txt = f"{confidence*100:.1f}% confidence" if confidence else ""
                st.markdown(f"""
                <div class="result-box result-bad">
                    <div class="result-title">🔴 Treatment Recommended</div>
                    <div class="result-conf">{conf_txt}</div>
                    <p class="result-desc">{explanation or "Based on your answers, seeking professional support is advised."}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                conf_txt = f"{confidence*100:.1f}% confidence" if confidence else ""
                st.markdown(f"""
                <div class="result-box result-ok">
                    <div class="result-title">🟢 Lower Immediate Risk</div>
                    <div class="result-conf">{conf_txt}</div>
                    <p class="result-desc">{explanation or "Your responses suggest a lower risk profile right now."}</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <div class="disclaimer">
                ⚠️ <strong>Disclaimer:</strong> This is for educational & awareness purposes only.
                It is <strong>not a medical diagnosis</strong>. Please consult a qualified mental health
                professional if you have concerns.
            </div>
            """, unsafe_allow_html=True)

        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. Railway might be waking up — try again in 10 seconds.")
        except requests.exceptions.ConnectionError:
            st.error("🌐 Can't reach the API. Check the URL in your Streamlit secrets.")
        except requests.exceptions.HTTPError as e:
            try:
                detail = response.json().get("detail", "")
            except Exception:
                detail = ""
            st.error(f"❌ API error {response.status_code}: {e}\n{detail}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")

# Debug expander at the bottom
with st.expander("🔧 Debug info"):
    st.code(f"API_URL = {API_URL}", language="text")
