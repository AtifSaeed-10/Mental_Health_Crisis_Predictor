import os
import streamlit as st
import requests

st.set_page_config(
    page_title="Mental Health Crisis Predictor",
    page_icon="����",
    layout="centered"
)

# --- CONFIG ---
# Priority: Streamlit secrets -> environment variable -> fallback URL
API_URL = (
    st.secrets.get("API_URL", None)
    if hasattr(st, "secrets")
    else None
) or os.getenv("API_URL") or "https://mentalhealthcrisispredictor-production.up.railway.app/predict"

REQUEST_TIMEOUT = 20  # seconds

# --- STYLES ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@200;300;400;600;700;800&family=Space+Mono:wght@400;700&display=swap');

:root {
    --ink:      #0b0d17;
    --muted:    #8a90a2;
    --accent1:  #7f5af0;
    --accent2:  #2cb67d;
}

* { font-family: 'Outfit', sans-serif; box-sizing: border-box; }

.stApp {
    background-color: var(--ink);
    color: white;
    min-height: 100vh;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 900px; }

.hero { text-align: center; padding: 2rem 1rem 1rem; }
.hero-brain {
    font-size: 4.5rem;
    display: block;
    margin-bottom: 0.6rem;
    filter: drop-shadow(0 0 25px rgba(127,90,240,0.55));
}
.hero h1 { font-size: 2.6rem; font-weight: 800; line-height: 1.1; margin: 0; }
.hero-sub { color: var(--muted); font-size: 0.95rem; margin-top: 0.4rem; }

.result-box {
    border-radius: 16px;
    padding: 1.3rem 1.2rem;
    margin-top: 1rem;
    text-align: center;
}
.result-ok {
    background: linear-gradient(135deg, #0a2419 0%, #0f1628 100%);
    border: 1px solid rgba(44,182,125,0.45);
}
.result-bad {
    background: linear-gradient(135deg, #1a0f3a 0%, #0f1628 100%);
    border: 1px solid rgba(127,90,240,0.45);
}
.result-title { font-size: 1.25rem; font-weight: 800; margin-bottom: 0.2rem; }
.result-desc  { color: #c7ccda; font-size: 0.95rem; }

.disclaimer {
    background: rgba(255,137,6,0.08);
    border: 1px solid rgba(255,137,6,0.25);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    margin-top: 1rem;
    font-size: 0.8rem;
    color: #d8b77f;
    line-height: 1.5;
}

div[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, #7f5af0 0%, #2cb67d 100%);
    color: white;
    font-size: 1rem;
    font-weight: 700;
    padding: 0.85rem 1.5rem;
    border: none;
    border-radius: 12px;
    margin-top: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# --- HERO ---
st.markdown("""
<div class="hero">
    <span class="hero-brain">🧠</span>
    <h1>Mental Health Crisis Predictor</h1>
    <p class="hero-sub">Text Classification via Railway FastAPI</p>
</div>
""", unsafe_allow_html=True)

# --- INPUT ---
st.markdown("### ✍️ Enter your text")
user_input = st.text_area(
    "Describe how you feel:",
    height=180,
    placeholder="e.g., I feel anxious, stressed, and unable to cope..."
)

# Optional debug visibility
with st.expander("API settings"):
    st.code(API_URL, language="text")

# --- PREDICT ---
if st.button("⚡ Analyze My Text"):
    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Connecting to Railway API... 🚀"):
            try:
                payload = {"text": user_input.strip()}
                response = requests.post(API_URL, json=payload, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()

                data = response.json()
                prediction = data.get("prediction")

                if prediction is None:
                    st.error("API response does not contain 'prediction'.")
                else:
                    pred_norm = str(prediction).strip().lower()
                    high_risk_labels = {"yes", "1", "true", "high risk", "treatment recommended"}

                    if pred_norm in high_risk_labels:
                        st.markdown(f"""
                        <div class="result-box result-bad">
                            <div class="result-title">🔴 Support Recommended</div>
                            <p class="result-desc">Prediction: <strong>{prediction}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-box result-ok">
                            <div class="result-title">🟢 Lower Immediate Risk</div>
                            <p class="result-desc">Prediction: <strong>{prediction}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("""
                    <div class="disclaimer">
                        ⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only
                        and is not a medical diagnosis.
                    </div>
                    """, unsafe_allow_html=True)

            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. Please try again.")
            except requests.exceptions.ConnectionError:
                st.error("🌐 Connection failed. Check API URL or network.")
            except requests.exceptions.HTTPError as e:
                backend_msg = ""
                try:
                    backend_msg = response.json().get("detail", "")
                except Exception:
                    pass
                st.error(f"❌ API HTTP error: {e}\n{backend_msg}")
            except ValueError:
                st.error("❌ API returned invalid JSON.")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
