import streamlit as st
import requests

st.set_page_config(
    page_title="Mental Health Crisis Predictor",
    page_icon="🧠",
    layout="centered"
)

# --- STYLES (same theme feel) ---
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
    color: var(--text);
    min-height: 100vh;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 4rem; }

.hero { text-align: center; padding: 2.5rem 1rem 1.5rem; }
.hero-brain { font-size: 5rem; display: block; margin-bottom: 0.8rem; filter: drop-shadow(0 0 30px rgba(127,90,240,0.6)); }
.hero h1 { font-size: 3rem; font-weight: 800; line-height: 1.1; margin: 0 0 0.5rem; }
.hero-sub { color: var(--muted); font-size: 1rem; font-weight: 300; margin-bottom: 0.5rem; }

.result-box {
    border-radius: 20px;
    padding: 2rem 1.6rem;
    margin-top: 1rem;
    text-align: center;
}
.result-ok {
    background: linear-gradient(135deg, #0a2419 0%, #0f1628 100%);
    border: 1px solid rgba(44,182,125,0.5);
}
.result-bad {
    background: linear-gradient(135deg, #1a0f3a 0%, #0f1628 100%);
    border: 1px solid rgba(127,90,240,0.5);
}
.result-title { font-size: 1.4rem; font-weight: 800; margin-bottom: 0.3rem; }
.result-desc  { color: var(--muted); font-size: 0.92rem; }

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
    font-size: 1.05rem;
    font-weight: 700;
    padding: 0.9rem 2rem;
    border: none;
    border-radius: 14px;
    cursor: pointer;
    margin-top: 1.2rem;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)

# --- HERO ---
st.markdown("""
<div class="hero">
    <span class="hero-brain">🧠</span>
    <h1>Mental Health<br>Crisis Predictor</h1>
    <p class="hero-sub">Powered by Railway API</p>
</div>
""", unsafe_allow_html=True)

# --- INPUT ---
st.markdown("### ✍️ Text Assessment")
user_input = st.text_area(
    "Describe how you are feeling:",
    height=180,
    placeholder="e.g., I feel very stressed, low, and disconnected from everyone..."
)

# --- API CONFIG ---
API_URL = "https://mentalhealthcrisispredictor-production.up.railway.app/predict"

# --- PREDICT ---
if st.button("⚡ Analyze My Text"):
    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Connecting to Railway API... 🚀"):
            try:
                payload = {"text": user_input}
                response = requests.post(API_URL, json=payload, timeout=15)
                response.raise_for_status()

                data = response.json()
                prediction = data.get("prediction")

                if prediction is None:
                    st.error("API response missing 'prediction' key.")
                else:
                    pred_str = str(prediction).strip().lower()

                    if pred_str in ["yes", "1", "true", "treatment recommended", "high risk"]:
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
                        ⚠️ <strong>Disclaimer:</strong> This is an educational tool and not a clinical diagnosis.
                        Please consult a qualified mental health professional for medical advice.
                    </div>
                    """, unsafe_allow_html=True)

            except requests.exceptions.Timeout:
                st.error("⏱️ Request timeout. Railway service may be sleeping. Please try again.")
            except requests.exceptions.ConnectionError:
                st.error("🌐 Could not connect to API. Check URL/network.")
            except requests.exceptions.HTTPError as e:
                st.error(f"❌ API HTTP error: {e}")
            except ValueError:
                st.error("❌ Invalid JSON received from API.")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
