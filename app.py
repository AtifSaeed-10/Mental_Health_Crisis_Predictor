import os
import streamlit as st
import requests
from typing import Dict, Tuple

st.set_page_config(
    page_title="MindVault · Mental Health Assessment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── API CONFIGURATION ────────────────────────────────────────────────────────
API_URL = (
    st.secrets.get("API_URL", None)
    if hasattr(st, "secrets")
    else None
) or os.getenv("API_URL") or "https://mentalhealthcrisispredictor-production.up.railway.app/predict"

REQUEST_TIMEOUT = 20

# ── PREMIUM CSS DESIGN ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

:root {
    --bg-primary: #fafbfc;
    --bg-secondary: #f5f7fa;
    --bg-tertiary: #eef2f7;
    --surface-light: rgba(255, 255, 255, 0.8);
    --surface-glass: rgba(255, 255, 255, 0.4);
    --text-primary: #0a0e27;
    --text-secondary: #6b7280;
    --text-tertiary: #9ca3af;
    --accent-primary: #3b82f6;
    --accent-secondary: #06b6d4;
    --accent-success: #10b981;
    --accent-danger: #ef4444;
    --accent-warning: #f59e0b;
    --border-light: rgba(0, 0, 0, 0.06);
    --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.04);
    --shadow-md: 0 8px 24px rgba(0, 0, 0, 0.06);
    --shadow-lg: 0 16px 48px rgba(0, 0, 0, 0.1);
}

html, body, .stApp {
    background: var(--bg-primary) !important;
    color: var(--text-primary);
    font-family: 'Sora', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
}

#MainMenu { display: none !important; }
footer { display: none !important; }
header { visibility: hidden !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

/* ── ANIMATED BACKGROUND ─────────────────────────────────────────────────── */
.animated-bg {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: -1;
    overflow: hidden;
    background: linear-gradient(135deg, #fafbfc 0%, #f5f7fa 50%, #f0f4f9 100%);
}

/* Neural Network Background */
.neural-bg {
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 30% 50%, rgba(59, 130, 246, 0.03) 0%, transparent 50%),
                radial-gradient(circle at 70% 50%, rgba(6, 182, 212, 0.03) 0%, transparent 50%),
                radial-gradient(circle at 50% 80%, rgba(16, 185, 129, 0.02) 0%, transparent 50%);
    animation: breathe 8s ease-in-out infinite;
}

@keyframes breathe {
    0%, 100% { 
        opacity: 0.4;
        filter: blur(100px);
    }
    50% { 
        opacity: 0.7;
        filter: blur(80px);
    }
}

/* Floating Healing Orbs */
.floating-sphere {
    position: absolute;
    border-radius: 50%;
    filter: blur(80px);
    opacity: 0.25;
    animation: float 20s ease-in-out infinite;
}

.sphere-1 {
    width: 400px;
    height: 400px;
    background: linear-gradient(135deg, #3b82f6, #06b6d4);
    top: -10%;
    right: -5%;
    animation: float 25s ease-in-out infinite, glow-pulse-1 4s ease-in-out infinite;
    animation-delay: 0s, 0s;
}

.sphere-2 {
    width: 300px;
    height: 300px;
    background: linear-gradient(135deg, #06b6d4, #10b981);
    bottom: -5%;
    left: 10%;
    animation: float 30s ease-in-out infinite reverse, glow-pulse-2 5s ease-in-out infinite;
    animation-delay: -7s, -2s;
}

.sphere-3 {
    width: 250px;
    height: 250px;
    background: linear-gradient(135deg, #10b981, #3b82f6);
    top: 50%;
    left: 5%;
    animation: float 22s ease-in-out infinite, glow-pulse-3 4.5s ease-in-out infinite;
    animation-delay: -14s, -1s;
}

@keyframes float {
    0%, 100% { transform: translate(0px, 0px) scale(1); }
    50% { transform: translate(30px, 30px) scale(1.05); }
}

@keyframes glow-pulse-1 {
    0%, 100% { filter: blur(80px) brightness(0.8); }
    50% { filter: blur(60px) brightness(1.1); }
}

@keyframes glow-pulse-2 {
    0%, 100% { filter: blur(80px) brightness(0.85); }
    50% { filter: blur(65px) brightness(1.15); }
}

@keyframes glow-pulse-3 {
    0%, 100% { filter: blur(80px) brightness(0.9); }
    50% { filter: blur(70px) brightness(1.2); }
}

/* Animated Wave Layer */
.wave-layer {
    position: absolute;
    inset: 0;
    background: 
        repeating-linear-gradient(
            0deg,
            rgba(59, 130, 246, 0.02) 0px,
            transparent 2px,
            transparent 20px,
            rgba(59, 130, 246, 0.02) 22px
        ),
        repeating-linear-gradient(
            90deg,
            rgba(6, 182, 212, 0.02) 0px,
            transparent 2px,
            transparent 20px,
            rgba(6, 182, 212, 0.02) 22px
        );
    animation: wave-drift 15s linear infinite;
    opacity: 0.4;
}

@keyframes wave-drift {
    0% { transform: translateX(0px) translateY(0px); }
    100% { transform: translateX(100px) translateY(50px); }
}

/* Light Beam Animation */
.light-beam {
    position: absolute;
    width: 2px;
    height: 200px;
    background: linear-gradient(180deg, rgba(59, 130, 246, 0) 0%, rgba(59, 130, 246, 0.4) 50%, rgba(59, 130, 246, 0) 100%);
    filter: blur(10px);
    animation: beam-sweep 6s ease-in-out infinite;
    pointer-events: none;
}

.beam-1 {
    top: 20%;
    left: 20%;
    animation: beam-sweep 6s ease-in-out infinite;
    animation-delay: 0s;
}

.beam-2 {
    top: 40%;
    right: 15%;
    background: linear-gradient(180deg, rgba(6, 182, 212, 0) 0%, rgba(6, 182, 212, 0.3) 50%, rgba(6, 182, 212, 0) 100%);
    animation: beam-sweep 7s ease-in-out infinite reverse;
    animation-delay: -2s;
}

.beam-3 {
    bottom: 20%;
    left: 50%;
    background: linear-gradient(180deg, rgba(16, 185, 129, 0) 0%, rgba(16, 185, 129, 0.3) 50%, rgba(16, 185, 129, 0) 100%);
    animation: beam-sweep 8s ease-in-out infinite;
    animation-delay: -4s;
}

@keyframes beam-sweep {
    0%, 100% { 
        opacity: 0;
        transform: scaleY(0.5) translateY(0px);
    }
    50% { 
        opacity: 1;
        transform: scaleY(1) translateY(100px);
    }
}

/* Particle System */
.particle {
    position: absolute;
    border-radius: 50%;
    pointer-events: none;
}

.particle-1 {
    width: 4px;
    height: 4px;
    background: rgba(59, 130, 246, 0.6);
    top: 20%;
    left: 30%;
    animation: float-particle 8s ease-in-out infinite;
    animation-delay: 0s;
}

.particle-2 {
    width: 3px;
    height: 3px;
    background: rgba(6, 182, 212, 0.5);
    top: 60%;
    right: 20%;
    animation: float-particle 10s ease-in-out infinite reverse;
    animation-delay: -3s;
}

.particle-3 {
    width: 5px;
    height: 5px;
    background: rgba(16, 185, 129, 0.4);
    bottom: 30%;
    left: 40%;
    animation: float-particle 9s ease-in-out infinite;
    animation-delay: -5s;
}

.particle-4 {
    width: 3px;
    height: 3px;
    background: rgba(59, 130, 246, 0.5);
    top: 70%;
    right: 30%;
    animation: float-particle 7s ease-in-out infinite reverse;
    animation-delay: -2s;
}

@keyframes float-particle {
    0%, 100% { 
        transform: translate(0px, 0px);
        opacity: 0.3;
    }
    50% { 
        transform: translate(60px, -60px);
        opacity: 0.8;
    }
}

/* Emotional State Glow */
.emotional-glow {
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at center, rgba(59, 130, 246, 0.05) 0%, transparent 70%);
    animation: emotional-pulse 10s ease-in-out infinite;
}

@keyframes emotional-pulse {
    0%, 100% { 
        opacity: 0.3;
        filter: blur(120px);
    }
    25% { 
        opacity: 0.5;
        filter: blur(100px);
    }
    50% { 
        opacity: 0.7;
        filter: blur(80px);
    }
    75% { 
        opacity: 0.4;
        filter: blur(110px);
    }
}

/* ── MAIN CONTAINER ──────────────────────────────────────────────────────── */
.main-container {
    max-width: 900px;
    margin: 0 auto;
    padding: 60px 24px 80px;
    position: relative;
    z-index: 1;
}

/* ── NAVIGATION BAR ──────────────────────────────────────────────────────── */
.nav-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 24px;
    background: var(--surface-glass);
    backdrop-filter: blur(20px);
    border: 1px solid var(--border-light);
    border-radius: 12px;
    margin-bottom: 48px;
    box-shadow: var(--shadow-sm);
}

.nav-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 20px;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.5px;
}

.nav-badge {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--accent-secondary);
    background: rgba(6, 182, 212, 0.1);
    border: 1px solid rgba(6, 182, 212, 0.3);
    padding: 6px 12px;
    border-radius: 20px;
}

/* ── HERO SECTION ────────────────────────────────────────────────────────── */
.hero {
    text-align: center;
    margin-bottom: 60px;
    animation: slideDownFade 0.8s cubic-bezier(0.16, 1, 0.3, 1);
}

@keyframes slideDownFade {
    from {
        opacity: 0;
        transform: translateY(-30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    background: rgba(59, 130, 246, 0.08);
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 24px;
    margin-bottom: 24px;
    font-size: 13px;
    font-weight: 600;
    color: var(--accent-primary);
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.5px;
}

.hero-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent-primary);
    animation: pulse-beat 2s ease-in-out infinite;
}

@keyframes pulse-beat {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(1.2); }
}

.hero h1 {
    font-size: clamp(2.2rem, 6vw, 3.5rem);
    font-weight: 700;
    line-height: 1.15;
    color: var(--text-primary);
    margin-bottom: 16px;
    letter-spacing: -0.02em;
}

.hero h1 span {
    background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero p {
    font-size: 16px;
    color: var(--text-secondary);
    line-height: 1.6;
    max-width: 500px;
    margin: 0 auto 32px;
    font-weight: 400;
}

.hero-stats {
    display: flex;
    justify-content: center;
    gap: 32px;
    flex-wrap: wrap;
    margin-top: 32px;
}

.stat-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
}

.stat-value {
    font-size: 22px;
    font-weight: 700;
    color: var(--accent-primary);
    font-family: 'Space Mono', monospace;
}

.stat-label {
    font-size: 12px;
    color: var(--text-tertiary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-family: 'Space Mono', monospace;
}

/* ── SECTION HEADER ──────────────────────────────────────────────────────── */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 40px 0 20px;
}

.section-number {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: linear-gradient(135deg, #3b82f6, #06b6d4);
    border-radius: 8px;
    color: white;
    font-weight: 700;
    font-size: 14px;
    font-family: 'Space Mono', monospace;
}

.section-title {
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    color: var(--text-primary);
    letter-spacing: 0.8px;
}

.section-line {
    flex: 1;
    height: 1px;
    background: var(--border-light);
}

/* ── FORM CARDS ──────────────────────────────────────────────────────────── */
.form-card {
    background: var(--surface-light);
    border: 1px solid var(--border-light);
    border-radius: 16px;
    padding: 32px;
    margin-bottom: 24px;
    box-shadow: var(--shadow-sm);
    backdrop-filter: blur(10px);
    animation: fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* ── SELECT BOX STYLING ──────────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    font-weight: 500 !important;
    height: 44px !important;
    transition: all 0.2s ease !important;
}

[data-testid="stSelectbox"] > div > div:hover {
    border-color: var(--accent-primary) !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
}

[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: var(--accent-primary) !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15) !important;
}

ul[role="listbox"] {
    background: var(--surface-light) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 12px !important;
    box-shadow: var(--shadow-lg) !important;
}

li[role="option"] {
    color: var(--text-primary) !important;
    font-size: 14px !important;
    padding: 12px 16px !important;
    transition: all 0.15s ease !important;
}

li[role="option"]:hover {
    background: var(--bg-secondary) !important;
    color: var(--accent-primary) !important;
}

[data-testid="stWidgetLabel"] > div > p {
    color: var(--text-primary) !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    margin-bottom: 8px !important;
    letter-spacing: 0.3px !important;
}

/* ── COLUMNS ─────────────────────────────────────────────────────────────── */
[data-testid="column"] {
    padding: 0 8px !important;
}

/* ── BUTTON STYLING ──────────────────────────────────────────────────────── */
[data-testid="stButton"] > button {
    width: 100% !important;
    background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%) !important;
    color: white !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    padding: 16px 32px !important;
    border: none !important;
    border-radius: 12px !important;
    height: auto !important;
    margin-top: 24px !important;
    letter-spacing: 0.3px !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    box-shadow: var(--shadow-md) !important;
    cursor: pointer !important;
    text-transform: uppercase;
}

[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 20px 40px rgba(59, 130, 246, 0.3) !important;
}

[data-testid="stButton"] > button:active {
    transform: translateY(0px) !important;
}

/* ── LOADING SPINNER ─────────────────────────────────────────────────────── */
[data-testid="stSpinner"] > div {
    border-color: var(--accent-primary) !important;
}

/* ── RESULT CARDS ────────────────────────────────────────────────────────── */
.result-container {
    animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    margin-top: 40px;
}

@keyframes slideUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.result-box {
    border-radius: 20px;
    padding: 48px 40px;
    text-align: center;
    position: relative;
    overflow: hidden;
    border: 1px solid var(--border-light);
    backdrop-filter: blur(10px);
}

.result-box::before {
    content: '';
    position: absolute;
    top: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 200%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.3), transparent);
}

.result-success {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.05) 0%, rgba(6, 182, 212, 0.05) 100%);
    border-color: rgba(16, 185, 129, 0.2);
}

.result-warning {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.05) 0%, rgba(245, 158, 11, 0.05) 100%);
    border-color: rgba(239, 68, 68, 0.2);
}

.result-icon {
    font-size: 64px;
    margin-bottom: 16px;
    animation: bounce 2s ease-in-out infinite;
}

@keyframes bounce {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-8px); }
}

.result-status {
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 8px;
    color: var(--accent-secondary);
}

.result-warning .result-status {
    color: var(--accent-danger);
}

.result-title {
    font-size: 32px;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 16px;
    line-height: 1.3;
}

.result-confidence {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    font-weight: 600;
    padding: 8px 16px;
    border-radius: 20px;
    margin-bottom: 20px;
    background: rgba(59, 130, 246, 0.1);
    color: var(--accent-primary);
    letter-spacing: 0.5px;
}

.result-warning .result-confidence {
    background: rgba(239, 68, 68, 0.1);
    color: var(--accent-danger);
}

.result-description {
    font-size: 15px;
    line-height: 1.7;
    color: var(--text-secondary);
    max-width: 450px;
    margin: 0 auto 32px;
}

.confidence-bar-container {
    max-width: 300px;
    margin: 0 auto;
}

.confidence-label {
    display: flex;
    justify-content: space-between;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: var(--text-tertiary);
    margin-bottom: 8px;
    letter-spacing: 0.5px;
}

.confidence-track {
    height: 6px;
    background: var(--bg-secondary);
    border-radius: 3px;
    overflow: hidden;
}

.confidence-fill {
    height: 100%;
    background: linear-gradient(90deg, #3b82f6, #06b6d4);
    border-radius: 3px;
    animation: fillBar 1s cubic-bezier(0.16, 1, 0.3, 1);
}

.result-warning .confidence-fill {
    background: linear-gradient(90deg, #ef4444, #f59e0b);
}

@keyframes fillBar {
    from { width: 0 !important; }
}

/* ── DISCLAIMER ──────────────────────────────────────────────────────────── */
.disclaimer {
    background: rgba(245, 158, 11, 0.05);
    border: 1px solid rgba(245, 158, 11, 0.2);
    border-radius: 12px;
    padding: 16px;
    margin-top: 24px;
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.6;
    display: flex;
    gap: 12px;
    align-items: flex-start;
}

.disclaimer-icon {
    flex-shrink: 0;
    font-size: 16px;
    margin-top: 2px;
}

/* ── FOOTER ──────────────────────────────────────────────────────────────── */
.footer {
    text-align: center;
    margin-top: 60px;
    padding-top: 40px;
    border-top: 1px solid var(--border-light);
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: var(--text-tertiary);
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ── DIVIDER ─────────────────────────────────────────────────────────────── */
.divider {
    height: 1px;
    background: var(--border-light);
    margin: 48px 0;
    position: relative;
}

.divider::after {
    content: '';
    position: absolute;
    left: 50%;
    top: -1px;
    transform: translateX(-50%);
    width: 80px;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-primary), transparent);
    border-radius: 2px;
}

/* ── RESPONSIVE ──────────────────────────────────────────────────────────── */
@media (max-width: 768px) {
    .main-container {
        padding: 40px 16px 60px;
    }
    
    .form-card {
        padding: 20px;
    }
    
    .nav-bar {
        padding: 12px 16px;
        margin-bottom: 32px;
    }
    
    .hero {
        margin-bottom: 40px;
    }
    
    .hero h1 {
        font-size: 2rem;
    }
    
    .hero-stats {
        gap: 20px;
    }
    
    .result-box {
        padding: 32px 20px;
    }
    
    .result-icon {
        font-size: 48px;
    }
    
    .result-title {
        font-size: 24px;
    }
}

/* ── EXPANDER STYLING ────────────────────────────────────────────────────── */
details {
    background: var(--surface-light) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 12px !important;
    margin-top: 24px !important;
}

summary {
    color: var(--text-secondary) !important;
    font-size: 13px !important;
    font-family: 'Space Mono', monospace !important;
    padding: 12px 16px !important;
}

</style>
""", unsafe_allow_html=True)

# ── ANIMATED BACKGROUND HTML ────────────────────────────────────────────────
st.markdown("""
<!-- ANIMATED BACKGROUND ──────────────────────────────────────────────────────── -->
<div class="animated-bg">
    <!-- Neural Network Breathing Layer -->
    <div class="neural-bg"></div>
    
    <!-- Emotional State Glow -->
    <div class="emotional-glow"></div>
    
    <!-- Wave Patterns -->
    <div class="wave-layer"></div>
    
    <!-- Floating Healing Orbs (Neural Nodes) -->
    <div class="floating-sphere sphere-1"></div>
    <div class="floating-sphere sphere-2"></div>
    <div class="floating-sphere sphere-3"></div>
    
    <!-- Light Beams (Hope and Clarity) -->
    <div class="light-beam beam-1"></div>
    <div class="light-beam beam-2"></div>
    <div class="light-beam beam-3"></div>
    
    <!-- Particle System (Neural Activity) -->
    <div class="particle particle-1"></div>
    <div class="particle particle-2"></div>
    <div class="particle particle-3"></div>
    <div class="particle particle-4"></div>
</div>
<div class="main-container">
""", unsafe_allow_html=True)

# ── NAVIGATION BAR ──────────────────────────────────────────────────────────
st.markdown("""
<div class="nav-bar">
    <div class="nav-logo">🧠 MindVault</div>
    <div class="nav-badge">AI Powered</div>
</div>
""", unsafe_allow_html=True)

# ── HERO SECTION ────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">
        <div class="hero-dot"></div>
        Personalized Mental Health Assessment
    </div>
    <h1>Understand Your <span>Mental State</span></h1>
    <p>
        Answer thoughtful questions about your lifestyle and mental patterns. 
        Our AI model provides evidence-based insights in seconds.
    </p>
    <div class="hero-stats">
        <div class="stat-item">
            <div class="stat-value">72%</div>
            <div class="stat-label">Model Accuracy</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">14</div>
            <div class="stat-label">Assessment Items</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">&lt;1s</div>
            <div class="stat-label">Prediction Time</div>
        </div>
    </div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ── FORM MAPPINGS ───────────────────────────────────────────────────────────
YESNO = {"Yes": 1, "No": 0}
YESNO_MAYBE = {"Yes": 1, "Maybe": 0.5, "No": 0}
GENDER_MAP = {"Female": 0, "Male": 1}
OCC_MAP = {"Corporate": 0, "Self-Employed": 1, "Student": 2, "Other": 3}
DAYS_MAP = {
    "Go out every day": 0,
    "1-14 days": 1,
    "15-30 days": 2,
    "31-60 days": 3,
    "More than 2 months": 4
}
MOOD_MAP = {"Low": 0, "Medium": 1, "High": 2}
CARE_MAP = {"No": 0, "Not sure": 0.5, "Yes": 1}

# ── SECTION 01: PERSONAL BACKGROUND ─────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-number">01</div>
    <div class="section-title">Personal Background</div>
    <div class="section-line"></div>
</div>
<div class="form-card">
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="medium")
with col1:
    gender = st.selectbox(
        "Gender",
        list(GENDER_MAP.keys()),
        key="gender_select"
    )
    occupation = st.selectbox(
        "Occupation",
        list(OCC_MAP.keys()),
        key="occupation_select"
    )
with col2:
    self_employed = st.selectbox(
        "Are you self-employed?",
        list(YESNO.keys()),
        key="self_emp_select"
    )
    family_history = st.selectbox(
        "Family history of mental illness?",
        list(YESNO.keys()),
        key="family_select"
    )

st.markdown('</div>', unsafe_allow_html=True)

# ── SECTION 02: STRESS & EMOTIONAL STATE ────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-number">02</div>
    <div class="section-title">Stress & Emotional State</div>
    <div class="section-line"></div>
</div>
<div class="form-card">
""", unsafe_allow_html=True)

col3, col4 = st.columns(2, gap="medium")
with col3:
    growing_stress = st.selectbox(
        "Growing stress lately?",
        list(YESNO_MAYBE.keys()),
        key="stress_select"
    )
    mood_swings = st.selectbox(
        "Mood swing intensity?",
        list(MOOD_MAP.keys()),
        key="mood_select"
    )
with col4:
    coping_struggles = st.selectbox(
        "Struggling to cope with daily life?",
        list(YESNO.keys()),
        key="coping_select"
    )
    changes_habits = st.selectbox(
        "Noticeable changes in habits?",
        list(YESNO_MAYBE.keys()),
        key="habits_select"
    )

st.markdown('</div>', unsafe_allow_html=True)

# ── SECTION 03: BEHAVIOR & SOCIAL LIFE ──────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-number">03</div>
    <div class="section-title">Behaviour & Social Life</div>
    <div class="section-line"></div>
</div>
<div class="form-card">
""", unsafe_allow_html=True)

col5, col6 = st.columns(2, gap="medium")
with col5:
    work_interest = st.selectbox(
        "Lost interest in work or hobbies?",
        list(YESNO_MAYBE.keys()),
        key="interest_select"
    )
    social_weakness = st.selectbox(
        "Feeling socially withdrawn?",
        list(YESNO_MAYBE.keys()),
        key="social_select"
    )
with col6:
    days_indoors = st.selectbox(
        "Days spent indoors (past month)?",
        list(DAYS_MAP.keys()),
        key="days_select"
    )

st.markdown('</div>', unsafe_allow_html=True)

# ── SECTION 04: AWARENESS & SUPPORT ─────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-number">04</div>
    <div class="section-title">Awareness & Support Access</div>
    <div class="section-line"></div>
</div>
<div class="form-card">
""", unsafe_allow_html=True)

col7, col8 = st.columns(2, gap="medium")
with col7:
    mh_history = st.selectbox(
        "Had mental health issues before?",
        list(YESNO_MAYBE.keys()),
        key="history_select"
    )
    mh_interview = st.selectbox(
        "Comfortable discussing MH in interviews?",
        list(YESNO_MAYBE.keys()),
        key="interview_select"
    )
with col8:
    care_options = st.selectbox(
        "Access to mental health care?",
        list(CARE_MAP.keys()),
        key="care_select"
    )

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── PREDICT BUTTON ──────────────────────────────────────────────────────────
predict_clicked = st.button("✨ Run Assessment", key="predict_button")

# ── PREDICTION LOGIC ────────────────────────────────────────────────────────
if predict_clicked:
    payload = {
        "Gender": GENDER_MAP[gender],
        "Occupation": OCC_MAP[occupation],
        "self_employed": YESNO[self_employed],
        "family_history": YESNO[family_history],
        "Days_Indoors": DAYS_MAP[days_indoors],
        "Growing_Stress": YESNO_MAYBE[growing_stress],
        "Changes_Habits": YESNO_MAYBE[changes_habits],
        "Mental_Health_History": YESNO_MAYBE[mh_history],
        "Mood_Swings": MOOD_MAP[mood_swings],
        "Coping_Struggles": YESNO[coping_struggles],
        "Work_Interest": YESNO_MAYBE[work_interest],
        "Social_Weakness": YESNO_MAYBE[social_weakness],
        "mental_health_interview": YESNO_MAYBE[mh_interview],
        "care_options": CARE_MAP[care_options],
    }

    with st.spinner("🔄 Analyzing your assessment..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            prediction = data.get("prediction", "Unknown")
            confidence = data.get("confidence", None)
            explanation = data.get("explanation", "")
            
            conf_pct = round(confidence * 100, 1) if confidence else 0
            conf_str = f"{conf_pct}%"
            bar_pct = max(conf_pct, 10)
            
            is_high_risk = str(prediction).strip().lower() in {"yes", "1", "true", "treatment recommended"}

            if is_high_risk:
                st.markdown(f"""
                <div class="result-container">
                    <div class="result-box result-warning">
                        <div class="result-icon">🔴</div>
                        <div class="result-status">Assessment Results</div>
                        <div class="result-title">Professional Support Recommended</div>
                        <div class="result-confidence">Confidence: {conf_str}</div>
                        <p class="result-description">
                            {explanation or 'Your assessment suggests that seeking professional mental health support could be beneficial for your wellbeing.'}
                        </p>
                        <div class="confidence-bar-container">
                            <div class="confidence-label">
                                <span>Confidence Score</span>
                                <span>{conf_str}</span>
                            </div>
                            <div class="confidence-track">
                                <div class="confidence-fill" style="width: {bar_pct}%"></div>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-container">
                    <div class="result-box result-success">
                        <div class="result-icon">🟢</div>
                        <div class="result-status">Assessment Results</div>
                        <div class="result-title">Positive Mental State</div>
                        <div class="result-confidence">Confidence: {conf_str}</div>
                        <p class="result-description">
                            {explanation or 'Your assessment indicates a healthy mental state. Continue prioritizing your wellbeing and self-care practices.'}
                        </p>
                        <div class="confidence-bar-container">
                            <div class="confidence-label">
                                <span>Confidence Score</span>
                                <span>{conf_str}</span>
                            </div>
                            <div class="confidence-track">
                                <div class="confidence-fill" style="width: {bar_pct}%"></div>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <div class="disclaimer">
                <span class="disclaimer-icon">⚠️</span>
                <span>
                    <strong>Important Disclaimer:</strong> MindVault is an educational assessment tool powered by machine learning. 
                    It is <strong>not a medical diagnosis</strong> and should never replace professional mental health consultation. 
                    If you're experiencing distress, please reach out to a licensed mental health professional or crisis support service.
                </span>
            </div>
            """, unsafe_allow_html=True)

        except requests.exceptions.Timeout:
            st.error("⏱️ The API is taking longer than expected. Please try again in a moment.")
        except requests.exceptions.ConnectionError:
            st.error("🌐 Unable to reach the assessment service. Please verify your connection and try again.")
        except requests.exceptions.HTTPError as e:
            try:
                detail = response.json()
                error_msg = detail.get("error", str(detail))
            except Exception:
                error_msg = response.text
            st.error(f"❌ Service Error: {error_msg}")
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {str(e)}")

# ── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    MindVault Assessment Platform &nbsp;·&nbsp; Built with ❤️ &nbsp;·&nbsp; Educational Use Only
</div>
</div>
""", unsafe_allow_html=True)

# ── DEBUG INFO (HIDDEN BY DEFAULT) ──────────────────────────────────────────
with st.expander("🔧 Debug Information"):
    st.code(f"API_URL: {API_URL}", language="text")
    st.info("This section is for development purposes only.")
