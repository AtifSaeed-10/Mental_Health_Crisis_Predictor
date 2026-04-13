import os
import streamlit as st
import requests

st.set_page_config(
    page_title="MindScan · Mental Health Predictor",
    page_icon="🧠",
    layout="centered"
)

API_URL = (
    st.secrets.get("API_URL", None)
    if hasattr(st, "secrets")
    else None
) or os.getenv("API_URL") or "https://mentalhealthcrisispredictor-production.up.railway.app/predict"

REQUEST_TIMEOUT = 20

# ── ULTRA PREMIUM CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Instrument+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@300;400&display=swap');

:root {
    --bg:           #060810;
    --surface:      #0d1117;
    --surface2:     #111827;
    --border:       rgba(255,255,255,0.06);
    --border-glow:  rgba(99,179,237,0.3);
    --cyan:         #63b3ed;
    --teal:         #4fd1c5;
    --rose:         #fc8181;
    --emerald:      #68d391;
    --violet:       #b794f4;
    --gold:         #f6e05e;
    --muted:        #4a5568;
    --muted2:       #718096;
    --text:         #e2e8f0;
    --text-bright:  #f7fafc;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    background: var(--bg) !important;
    color: var(--text);
    font-family: 'Instrument Sans', sans-serif;
}

#MainMenu, footer, header,
div[data-testid="stToolbar"],
div[data-testid="stDecoration"] { visibility: hidden !important; height: 0 !important; }

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── ANIMATED BACKGROUND ── */
.bg-layer {
    position: fixed; inset: 0; z-index: 0; overflow: hidden; pointer-events: none;
}
.orb {
    position: absolute; border-radius: 50%;
    filter: blur(120px); opacity: 0.12;
    animation: drift 20s ease-in-out infinite alternate;
}
.orb-1 { width: 600px; height: 600px; background: #63b3ed; top: -200px; left: -200px; animation-delay: 0s; }
.orb-2 { width: 500px; height: 500px; background: #b794f4; bottom: -150px; right: -150px; animation-delay: -7s; }
.orb-3 { width: 400px; height: 400px; background: #4fd1c5; top: 40%; left: 50%; transform: translate(-50%,-50%); animation-delay: -13s; }

@keyframes drift {
    0%   { transform: translate(0, 0) scale(1); }
    100% { transform: translate(40px, 30px) scale(1.08); }
}

.grid-texture {
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background-image:
        linear-gradient(rgba(99,179,237,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(99,179,237,0.03) 1px, transparent 1px);
    background-size: 60px 60px;
}

/* ── MAIN WRAPPER ── */
.main-wrapper {
    position: relative; z-index: 1;
    max-width: 760px; margin: 0 auto;
    padding: 0 1.5rem 4rem;
}

/* ── TOP BAR ── */
.topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1.4rem 0 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0;
}
.topbar-logo {
    display: flex; align-items: center; gap: 0.5rem;
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem; font-weight: 700;
    color: var(--text-bright); letter-spacing: -0.01em;
}
.topbar-logo-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--cyan); box-shadow: 0 0 12px var(--cyan);
}
.topbar-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; color: var(--muted2);
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    padding: 0.3rem 0.7rem; border-radius: 20px;
    letter-spacing: 0.08em;
}

/* ── HERO ── */
.hero {
    text-align: center;
    padding: 3.5rem 0 2.5rem;
}
.hero-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--cyan);
    background: rgba(99,179,237,0.08);
    border: 1px solid rgba(99,179,237,0.2);
    padding: 0.35rem 0.9rem; border-radius: 20px;
    margin-bottom: 1.6rem;
}
.hero-badge-dot {
    width: 5px; height: 5px; border-radius: 50%;
    background: var(--cyan); box-shadow: 0 0 8px var(--cyan);
    animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.7); }
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.4rem, 5vw, 3.4rem);
    font-weight: 900; line-height: 1.08;
    color: var(--text-bright);
    letter-spacing: -0.03em;
    margin-bottom: 0.2rem;
}
.hero h1 span {
    background: linear-gradient(135deg, var(--cyan) 0%, var(--teal) 40%, var(--violet) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 0.95rem; color: var(--muted2); line-height: 1.6;
    max-width: 420px; margin: 1rem auto 0;
    font-weight: 300;
}

/* ── STATS ROW ── */
.stats-row {
    display: flex; gap: 1rem; margin: 2rem 0 0.5rem;
    justify-content: center; flex-wrap: wrap;
}
.stat-chip {
    display: flex; align-items: center; gap: 0.4rem;
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 10px; padding: 0.55rem 1rem;
    font-size: 0.78rem; color: var(--muted2);
    font-family: 'JetBrains Mono', monospace;
}
.stat-chip b { color: var(--text); font-size: 0.88rem; }

/* ── DIVIDER ── */
.divider {
    height: 1px; background: var(--border);
    margin: 2rem 0; position: relative;
}
.divider::after {
    content: '';
    position: absolute; left: 50%; top: -1px;
    transform: translateX(-50%);
    width: 60px; height: 2px;
    background: linear-gradient(90deg, var(--cyan), var(--violet));
    border-radius: 2px;
}

/* ── SECTION HEADER ── */
.section-head {
    display: flex; align-items: center; gap: 0.8rem;
    margin: 1.8rem 0 1rem;
}
.section-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem; font-weight: 400;
    color: var(--cyan);
    background: rgba(99,179,237,0.1);
    border: 1px solid rgba(99,179,237,0.2);
    width: 28px; height: 28px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.section-title {
    font-size: 0.8rem; font-weight: 600;
    color: var(--text); letter-spacing: 0.06em;
    text-transform: uppercase;
}
.section-line { flex: 1; height: 1px; background: var(--border); }

/* ── CARD ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.4rem 1.4rem 1rem;
    margin-bottom: 1rem;
    position: relative; overflow: hidden;
}
.card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,179,237,0.4), transparent);
}

/* ── WIDGET OVERRIDES ── */
div[data-testid="stSelectbox"] > div,
div[data-baseweb="select"] > div {
    background: var(--surface2) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    transition: border-color 0.2s !important;
}
div[data-testid="stSelectbox"] > div:hover {
    border-color: rgba(99,179,237,0.4) !important;
}
label[data-testid="stWidgetLabel"] > div > p {
    color: var(--muted2) !important;
    font-size: 0.82rem !important;
    font-family: 'Instrument Sans', sans-serif !important;
    font-weight: 500 !important;
    margin-bottom: 0.3rem !important;
}
ul[role="listbox"] {
    background: #1a2035 !important;
    border: 1px solid var(--border-glow) !important;
    border-radius: 10px !important;
}
li[role="option"] { color: var(--text) !important; font-size: 0.88rem !important; }
li[role="option"]:hover { background: rgba(99,179,237,0.1) !important; }

/* ── BUTTON ── */
div[data-testid="stButton"] > button {
    width: 100% !important;
    background: linear-gradient(135deg, #2d6a9f 0%, #1a4a7a 50%, #2d3a7a 100%) !important;
    color: white !important;
    font-family: 'Instrument Sans', sans-serif !important;
    font-size: 1rem !important; font-weight: 600 !important;
    padding: 1rem 2rem !important;
    border: 1px solid rgba(99,179,237,0.4) !important;
    border-radius: 14px !important;
    margin-top: 0.5rem !important;
    letter-spacing: 0.04em !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 0 30px rgba(99,179,237,0.15), inset 0 1px 0 rgba(255,255,255,0.1) !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 50px rgba(99,179,237,0.3), inset 0 1px 0 rgba(255,255,255,0.15) !important;
    border-color: rgba(99,179,237,0.7) !important;
}

/* ── RESULT ── */
.result-wrap {
    animation: slideUp 0.5s cubic-bezier(0.16, 1, 0.3, 1);
    margin-top: 1.5rem;
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-box {
    border-radius: 20px; padding: 2rem 1.8rem;
    text-align: center; position: relative; overflow: hidden;
}
.result-box::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 50% 0%, var(--glow-color) 0%, transparent 65%);
    opacity: 0.15;
}
.result-ok  { background: linear-gradient(145deg, #071a12 0%, #0a1628 100%); border: 1px solid rgba(104,211,145,0.35); --glow-color: #68d391; }
.result-bad { background: linear-gradient(145deg, #1a0a14 0%, #1a0c28 100%); border: 1px solid rgba(252,129,129,0.35); --glow-color: #fc8181; }
.result-icon { font-size: 2.8rem; display: block; margin-bottom: 0.6rem; }
.result-status {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; letter-spacing: 0.18em; text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.result-ok  .result-status { color: var(--emerald); }
.result-bad .result-status { color: var(--rose); }
.result-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.7rem; font-weight: 700;
    color: var(--text-bright); margin-bottom: 0.6rem; line-height: 1.2;
}
.result-conf {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;
    padding: 0.3rem 0.8rem; border-radius: 20px; margin-bottom: 0.8rem;
}
.result-ok  .result-conf { background: rgba(104,211,145,0.12); color: var(--emerald); border: 1px solid rgba(104,211,145,0.25); }
.result-bad .result-conf { background: rgba(252,129,129,0.12); color: var(--rose);    border: 1px solid rgba(252,129,129,0.25); }
.result-desc { color: #a0aec0; font-size: 0.92rem; line-height: 1.6; max-width: 380px; margin: 0 auto; }
.conf-bar-wrap { margin: 1.2rem auto 0; max-width: 300px; }
.conf-bar-label {
    display: flex; justify-content: space-between;
    font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: var(--muted2);
    margin-bottom: 0.4rem;
}
.conf-bar-track { height: 4px; background: rgba(255,255,255,0.06); border-radius: 4px; overflow: hidden; }
.conf-bar-fill  { height: 100%; border-radius: 4px; animation: barGrow 0.8s cubic-bezier(0.16,1,0.3,1); }
.result-ok  .conf-bar-fill { background: linear-gradient(90deg, #68d391, #4fd1c5); }
.result-bad .conf-bar-fill { background: linear-gradient(90deg, #fc8181, #f6ad55); }
@keyframes barGrow { from { width: 0 !important; } }

/* ── DISCLAIMER ── */
.disclaimer {
    background: rgba(246,224,94,0.04); border: 1px solid rgba(246,224,94,0.15);
    border-radius: 12px; padding: 0.9rem 1.1rem; margin-top: 1.2rem;
    font-size: 0.78rem; color: #b7a56a; line-height: 1.6;
    display: flex; gap: 0.6rem; align-items: flex-start;
}
.disclaimer-icon { flex-shrink: 0; font-size: 0.9rem; margin-top: 0.05rem; }

/* ── FOOTER ── */
.footer {
    text-align: center; margin-top: 3rem; padding-top: 1.5rem;
    border-top: 1px solid var(--border);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; color: var(--muted);
    letter-spacing: 0.06em;
}
div[data-testid="column"] { padding: 0 0.4rem !important; }
details { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; }
summary { color: var(--muted2) !important; font-size: 0.78rem !important; font-family: 'JetBrains Mono', monospace !important; }
</style>
""", unsafe_allow_html=True)

# ── BACKGROUND ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="bg-layer">
    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>
    <div class="orb orb-3"></div>
</div>
<div class="grid-texture"></div>
<div class="main-wrapper">
""", unsafe_allow_html=True)

# ── TOP BAR ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="topbar-logo">
        <div class="topbar-logo-dot"></div>
        MindScan
    </div>
    <div class="topbar-tag">v2.0 &nbsp;·&nbsp; ML POWERED</div>
</div>
""", unsafe_allow_html=True)

# ── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">
        <div class="hero-badge-dot"></div>
        AI Mental Health Assessment
    </div>
    <h1>Know Your <span>Mind's</span><br>True State</h1>
    <p class="hero-sub">
        Answer 14 questions about your lifestyle and mental patterns.
        Our ensemble ML model gives you a data-driven health signal in seconds.
    </p>
</div>
<div class="stats-row">
    <div class="stat-chip">🎯 <b>71.5%</b>&nbsp;accuracy</div>
    <div class="stat-chip">⚡ <b>~1s</b>&nbsp;prediction</div>
    <div class="stat-chip">🔒 <b>Private</b>&nbsp;· no data stored</div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ── MAPS ─────────────────────────────────────────────────────────────────────
YESNO       = {"Yes": 1, "No": 0}
YESNO_MAYBE = {"Yes": 1, "Maybe": 0.5, "No": 0}
GENDER_MAP  = {"Female": 0, "Male": 1}
OCC_MAP     = {"Corporate": 0, "Self-Employed": 1, "Student": 2, "Other": 3}
DAYS_MAP    = {"Go out every day": 0, "1-14 days": 1, "15-30 days": 2,
               "31-60 days": 3, "More than 2 months": 4}
MOOD_MAP    = {"Low": 0, "Medium": 1, "High": 2}
CARE_MAP    = {"No": 0, "Not sure": 0.5, "Yes": 1}

# ── SECTION 1 ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-head">
    <div class="section-num">01</div>
    <div class="section-title">Personal Background</div>
    <div class="section-line"></div>
</div>
<div class="card">
""", unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    gender     = st.selectbox("Gender", list(GENDER_MAP.keys()))
    occupation = st.selectbox("Occupation", list(OCC_MAP.keys()))
with c2:
    self_employed  = st.selectbox("Are you self-employed?", list(YESNO.keys()))
    family_history = st.selectbox("Family history of mental illness?", list(YESNO.keys()))
st.markdown('</div>', unsafe_allow_html=True)

# ── SECTION 2 ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-head">
    <div class="section-num">02</div>
    <div class="section-title">Stress &amp; Emotional State</div>
    <div class="section-line"></div>
</div>
<div class="card">
""", unsafe_allow_html=True)
c3, c4 = st.columns(2)
with c3:
    growing_stress   = st.selectbox("Growing stress lately?", list(YESNO_MAYBE.keys()))
    mood_swings      = st.selectbox("Mood swing intensity?", list(MOOD_MAP.keys()))
with c4:
    coping_struggles = st.selectbox("Struggling to cope with daily life?", list(YESNO.keys()))
    changes_habits   = st.selectbox("Noticeable changes in habits?", list(YESNO_MAYBE.keys()))
st.markdown('</div>', unsafe_allow_html=True)

# ── SECTION 3 ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-head">
    <div class="section-num">03</div>
    <div class="section-title">Behaviour &amp; Social Life</div>
    <div class="section-line"></div>
</div>
<div class="card">
""", unsafe_allow_html=True)
c5, c6 = st.columns(2)
with c5:
    work_interest   = st.selectbox("Lost interest in work or hobbies?", list(YESNO_MAYBE.keys()))
    social_weakness = st.selectbox("Feeling socially withdrawn?", list(YESNO_MAYBE.keys()))
with c6:
    days_indoors    = st.selectbox("Days spent indoors (past month)?", list(DAYS_MAP.keys()))
st.markdown('</div>', unsafe_allow_html=True)

# ── SECTION 4 ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-head">
    <div class="section-num">04</div>
    <div class="section-title">Awareness &amp; Support Access</div>
    <div class="section-line"></div>
</div>
<div class="card">
""", unsafe_allow_html=True)
c7, c8 = st.columns(2)
with c7:
    mh_history   = st.selectbox("Had mental health issues before?", list(YESNO_MAYBE.keys()))
    mh_interview = st.selectbox("Comfortable discussing MH in interviews?", list(YESNO_MAYBE.keys()))
with c8:
    care_options = st.selectbox("Access to mental health care?", list(CARE_MAP.keys()))
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── BUTTON ───────────────────────────────────────────────────────────────────
predict_clicked = st.button("⚡  Run Mental Health Analysis")

# ── LOGIC ────────────────────────────────────────────────────────────────────
if predict_clicked:
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

    with st.spinner("Analyzing your responses via Railway AI..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data        = response.json()
            prediction  = data.get("prediction", "Unknown")
            confidence  = data.get("confidence", None)
            explanation = data.get("explanation", "")
            conf_pct    = round(confidence * 100, 1) if confidence else None
            conf_w      = f"{conf_pct}%" if conf_pct else "—"
            bar_w       = f"{conf_pct}%" if conf_pct else "60%"
            is_high     = str(prediction).strip().lower() in {"yes", "1", "true", "treatment recommended"}

            if is_high:
                st.markdown(f"""
                <div class="result-wrap">
                  <div class="result-box result-bad">
                    <span class="result-icon">🔴</span>
                    <div class="result-status">Assessment Complete</div>
                    <div class="result-title">Treatment Recommended</div>
                    <div class="result-conf">Model confidence &middot; {conf_w}</div>
                    <p class="result-desc">{explanation or "Your response pattern suggests professional mental health support would be beneficial."}</p>
                    <div class="conf-bar-wrap">
                      <div class="conf-bar-label"><span>confidence</span><span>{conf_w}</span></div>
                      <div class="conf-bar-track"><div class="conf-bar-fill" style="width:{bar_w}"></div></div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-wrap">
                  <div class="result-box result-ok">
                    <span class="result-icon">🟢</span>
                    <div class="result-status">Assessment Complete</div>
                    <div class="result-title">Lower Immediate Risk</div>
                    <div class="result-conf">Model confidence &middot; {conf_w}</div>
                    <p class="result-desc">{explanation or "Your responses indicate a relatively healthy mental state. Keep prioritising self-care."}</p>
                    <div class="conf-bar-wrap">
                      <div class="conf-bar-label"><span>confidence</span><span>{conf_w}</span></div>
                      <div class="conf-bar-track"><div class="conf-bar-fill" style="width:{bar_w}"></div></div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <div class="disclaimer">
                <span class="disclaimer-icon">⚠️</span>
                <span><strong>Medical Disclaimer:</strong> MindScan is an educational ML demonstration.
                It is <strong>not a clinical diagnosis</strong> and should not replace consultation
                with a licensed mental health professional. If you are in distress, please seek help immediately.</span>
            </div>
            """, unsafe_allow_html=True)

        except requests.exceptions.Timeout:
            st.error("⏱️ Railway is waking up — please try again in 15 seconds.")
        except requests.exceptions.ConnectionError:
            st.error("🌐 Cannot reach Railway API. Check your Streamlit secrets → API_URL.")
        except requests.exceptions.HTTPError as e:
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            st.error(f"❌ API {response.status_code} error: {detail}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")

# ── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    MINDSCAN &nbsp;·&nbsp; POWERED BY RAILWAY + STREAMLIT &nbsp;·&nbsp; NOT FOR CLINICAL USE
</div>
</div>
""", unsafe_allow_html=True)

with st.expander("🔧 Debug"):
    st.code(f"API_URL = {API_URL}", language="text")
