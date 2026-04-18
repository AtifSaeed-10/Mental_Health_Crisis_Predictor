import os
import streamlit as st
import requests
import math

st.set_page_config(
    page_title="MindVault · Mental Health Assessment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

API_URL = (
    st.secrets.get("API_URL", None)
    if hasattr(st, "secrets")
    else None
) or os.getenv("API_URL") or "https://mentalhealthcrisispredictor-production.up.railway.app/predict"

REQUEST_TIMEOUT = 20

# ── CUTTING EDGE AESTHETIC CSS ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

:root {
    --text-primary: #0a0e27;
    --text-secondary: #6b7280;
    --text-tertiary: #9ca3af;
    --accent-primary: #3b82f6;
    --accent-secondary: #06b6d4;
    --accent-success: #10b981;
    --accent-danger: #ef4444;
}

html, body, .stApp {
    background: #000000 !important;
    color: var(--text-primary);
    font-family: 'Sora', sans-serif;
    overflow-x: hidden;
}

/* ──────────────────────────────────────────────────────────────────────────── */
/* ANIMATED BACKGROUND - CUTTING EDGE */
/* ──────────────────────────────────────────────────────────────────────────── */

.animated-bg {
    position: fixed;
    inset: 0;
    width: 100%;
    height: 100%;
    z-index: -1;
    overflow: hidden;
    background: #0a0a0a;
}

/* Animated Gradient Mesh */
.gradient-mesh {
    position: absolute;
    inset: 0;
    background: linear-gradient(45deg, #1a1a2e 0%, #16213e 25%, #0f3460 50%, #1a1a2e 75%, #0a0e27 100%);
    background-size: 400% 400%;
    animation: gradient-shift 15s ease infinite;
}

@keyframes gradient-shift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Organic Blob Animations */
.blob {
    position: absolute;
    border-radius: 40% 60% 70% 30% / 40% 50% 60% 50%;
    filter: blur(100px);
    opacity: 0.15;
    animation: morphing 8s ease-in-out infinite;
}

.blob-1 {
    width: 500px;
    height: 500px;
    background: linear-gradient(135deg, #3b82f6, #06b6d4);
    top: -10%;
    left: 10%;
    animation: morphing 8s ease-in-out infinite, float-blob-1 20s ease-in-out infinite;
}

.blob-2 {
    width: 400px;
    height: 400px;
    background: linear-gradient(135deg, #06b6d4, #10b981);
    top: 30%;
    right: 5%;
    animation: morphing 10s ease-in-out infinite reverse, float-blob-2 25s ease-in-out infinite;
}

.blob-3 {
    width: 350px;
    height: 350px;
    background: linear-gradient(135deg, #10b981, #3b82f6);
    bottom: 10%;
    left: 50%;
    animation: morphing 9s ease-in-out infinite, float-blob-3 22s ease-in-out infinite;
}

.blob-4 {
    width: 300px;
    height: 300px;
    background: linear-gradient(135deg, #f59e0b, #06b6d4);
    bottom: 20%;
    right: 10%;
    animation: morphing 7s ease-in-out infinite reverse, float-blob-4 18s ease-in-out infinite;
}

@keyframes morphing {
    0%, 100% { border-radius: 40% 60% 70% 30% / 40% 50% 60% 50%; }
    50% { border-radius: 30% 70% 70% 30% / 30% 30% 70% 70%; }
}

@keyframes float-blob-1 {
    0%, 100% { transform: translate(0px, 0px) scale(1); }
    50% { transform: translate(50px, -50px) scale(1.1); }
}

@keyframes float-blob-2 {
    0%, 100% { transform: translate(0px, 0px) scale(1); }
    50% { transform: translate(-60px, 40px) scale(1.08); }
}

@keyframes float-blob-3 {
    0%, 100% { transform: translate(0px, 0px) scale(1); }
    50% { transform: translate(40px, -40px) scale(1.05); }
}

@keyframes float-blob-4 {
    0%, 100% { transform: translate(0px, 0px) scale(1); }
    50% { transform: translate(-40px, -30px) scale(1.1); }
}

/* Animated Light Streaks */
.light-streak {
    position: absolute;
    background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.4), transparent);
    filter: blur(40px);
    pointer-events: none;
}

.streak-1 {
    width: 800px;
    height: 2px;
    top: 20%;
    left: -400px;
    animation: streak-move-1 8s ease-in-out infinite;
}

.streak-2 {
    width: 600px;
    height: 2px;
    top: 50%;
    right: -300px;
    background: linear-gradient(90deg, transparent, rgba(6, 182, 212, 0.3), transparent);
    animation: streak-move-2 10s ease-in-out infinite reverse;
}

.streak-3 {
    width: 700px;
    height: 2px;
    bottom: 30%;
    left: -350px;
    background: linear-gradient(90deg, transparent, rgba(16, 185, 129, 0.3), transparent);
    animation: streak-move-3 9s ease-in-out infinite;
}

@keyframes streak-move-1 {
    0%, 100% { transform: translateX(0px); }
    50% { transform: translateX(400px); }
}

@keyframes streak-move-2 {
    0%, 100% { transform: translateX(0px); }
    50% { transform: translateX(-300px); }
}

@keyframes streak-move-3 {
    0%, 100% { transform: translateX(0px); }
    50% { transform: translateX(350px); }
}

/* Pulsing Aura */
.aura {
    position: absolute;
    border-radius: 50%;
    filter: blur(150px);
    pointer-events: none;
}

.aura-1 {
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, rgba(59, 130, 246, 0.2), transparent);
    top: 15%;
    left: 20%;
    animation: pulse-aura 6s ease-in-out infinite;
}

.aura-2 {
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(6, 182, 212, 0.15), transparent);
    bottom: 15%;
    right: 15%;
    animation: pulse-aura 7s ease-in-out infinite reverse;
}

@keyframes pulse-aura {
    0%, 100% { transform: scale(0.8); opacity: 0.3; }
    50% { transform: scale(1.1); opacity: 0.6; }
}

/* Noise Texture Overlay */
.noise-overlay {
    position: absolute;
    inset: 0;
    background-image: 
        url("data:image/svg+xml,%3Csvg viewBox='0 0 400 400' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' result='noise'/%3E%3C/filter%3E%3Crect width='400' height='400' fill='%23fff' filter='url(%23noiseFilter)' opacity='0.02'/%3E%3C/svg%3E");
    background-size: 200px 200px;
    opacity: 0.4;
    animation: noise-drift 20s linear infinite;
}

@keyframes noise-drift {
    0% { transform: translate(0px, 0px); }
    100% { transform: translate(200px, 200px); }
}

#MainMenu { display: none !important; }
footer { display: none !important; }
header { visibility: hidden !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

/* ──────────────────────────────────────────────────────────────────────────── */
/* MAIN CONTAINER - FLOWING LAYOUT */
/* ──────────────────────────────────────────────────────────────────────────── */

.main-wrapper {
    position: relative;
    z-index: 1;
    max-width: 1000px;
    margin: 0 auto;
    padding: 80px 40px 120px;
    background: linear-gradient(180deg, rgba(10, 10, 10, 0.3) 0%, rgba(10, 10, 10, 0.5) 100%);
}

/* Navigation */
.nav-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 20px 0;
    margin-bottom: 60px;
    border-bottom: 1px solid rgba(59, 130, 246, 0.1);
    animation: slideDown 0.8s cubic-bezier(0.16, 1, 0.3, 1);
}

@keyframes slideDown {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
}

.nav-logo {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 22px;
    font-weight: 800;
    background: linear-gradient(135deg, #3b82f6, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
}

.nav-badge {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #06b6d4;
    background: rgba(6, 182, 212, 0.1);
    border: 1px solid rgba(6, 182, 212, 0.2);
    padding: 8px 16px;
    border-radius: 40px;
    transition: all 0.3s ease;
}

.nav-badge:hover {
    background: rgba(6, 182, 212, 0.15);
    box-shadow: 0 0 20px rgba(6, 182, 212, 0.2);
}

/* Hero Section */
.hero {
    text-align: center;
    margin-bottom: 80px;
    animation: fadeInUp 1s cubic-bezier(0.16, 1, 0.3, 1);
    animation-delay: 0.1s;
    animation-fill-mode: both;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
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
    padding: 12px 20px;
    background: rgba(59, 130, 246, 0.1);
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 50px;
    margin-bottom: 32px;
    font-size: 12px;
    font-weight: 700;
    color: #3b82f6;
    font-family: 'Space Mono', monospace;
    letter-spacing: 1px;
    transition: all 0.3s ease;
}

.hero-badge:hover {
    background: rgba(59, 130, 246, 0.15);
    box-shadow: 0 0 30px rgba(59, 130, 246, 0.2);
    border-color: rgba(59, 130, 246, 0.4);
}

.hero-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #3b82f6;
    box-shadow: 0 0 15px #3b82f6;
    animation: pulse-dot 2s ease-in-out infinite;
}

@keyframes pulse-dot {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.3); opacity: 0.5; }
}

.hero h1 {
    font-size: clamp(2.5rem, 8vw, 4.5rem);
    font-weight: 800;
    line-height: 1.1;
    color: white;
    margin-bottom: 20px;
    letter-spacing: -2px;
    word-spacing: 100vw;
}

.hero h1 span {
    background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 50%, #10b981 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradient-shift 8s ease infinite;
    background-size: 200% 200%;
}

.hero p {
    font-size: 18px;
    color: #9ca3af;
    line-height: 1.8;
    max-width: 600px;
    margin: 24px auto 0;
    font-weight: 300;
    animation: fadeInUp 1s cubic-bezier(0.16, 1, 0.3, 1) 0.2s both;
}

.hero-stats {
    display: flex;
    justify-content: center;
    gap: 60px;
    margin-top: 48px;
    flex-wrap: wrap;
    animation: fadeInUp 1s cubic-bezier(0.16, 1, 0.3, 1) 0.3s both;
}

.stat-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    transition: transform 0.3s ease;
}

.stat-item:hover {
    transform: translateY(-5px);
}

.stat-value {
    font-size: 28px;
    font-weight: 800;
    background: linear-gradient(135deg, #3b82f6, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-family: 'Space Mono', monospace;
}

.stat-label {
    font-size: 11px;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-family: 'Space Mono', monospace;
    font-weight: 600;
}

/* Divider */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.3), transparent);
    margin: 80px 0;
    animation: fadeInUp 1s cubic-bezier(0.16, 1, 0.3, 1) 0.4s both;
}

/* Section Headers - Floating */
.section-header {
    display: flex;
    align-items: center;
    gap: 16px;
    margin: 60px 0 40px;
    animation: fadeInUp 1s cubic-bezier(0.16, 1, 0.3, 1) both;
}

.section-header:nth-of-type(2) { animation-delay: 0.1s; }
.section-header:nth-of-type(3) { animation-delay: 0.2s; }
.section-header:nth-of-type(4) { animation-delay: 0.3s; }
.section-header:nth-of-type(5) { animation-delay: 0.4s; }

.section-number {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 40px;
    height: 40px;
    background: linear-gradient(135deg, #3b82f6, #06b6d4);
    border-radius: 12px;
    color: white;
    font-weight: 800;
    font-size: 16px;
    font-family: 'Space Mono', monospace;
    flex-shrink: 0;
    box-shadow: 0 8px 24px rgba(59, 130, 246, 0.3);
}

.section-title {
    font-size: 16px;
    font-weight: 700;
    text-transform: uppercase;
    color: white;
    letter-spacing: 1.5px;
}

.section-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(59, 130, 246, 0.2), transparent);
}

/* Form Container - Organic Flow */
.form-section {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.05) 0%, rgba(6, 182, 212, 0.03) 100%);
    border: 1px solid rgba(59, 130, 246, 0.1);
    border-radius: 24px;
    padding: 48px;
    margin-bottom: 32px;
    backdrop-filter: blur(20px);
    animation: fadeInUp 1s cubic-bezier(0.16, 1, 0.3, 1) both;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.form-section::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.4), transparent);
}

.form-section:hover {
    border-color: rgba(59, 130, 246, 0.2);
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.08) 0%, rgba(6, 182, 212, 0.05) 100%);
    box-shadow: 0 20px 60px rgba(59, 130, 246, 0.1);
}

/* Input Styling */
[data-testid="stSelectbox"] > div > div {
    background: rgba(30, 30, 50, 0.8) !important;
    border: 1px solid rgba(59, 130, 246, 0.2) !important;
    border-radius: 14px !important;
    color: white !important;
    font-weight: 500 !important;
    height: 48px !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    backdrop-filter: blur(20px) !important;
}

[data-testid="stSelectbox"] > div > div:hover {
    border-color: #3b82f6 !important;
    background: rgba(30, 30, 50, 0.95) !important;
    box-shadow: 0 0 30px rgba(59, 130, 246, 0.2) !important;
}

[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: #06b6d4 !important;
    box-shadow: 0 0 40px rgba(6, 182, 212, 0.3) !important;
}

ul[role="listbox"] {
    background: rgba(20, 20, 35, 0.95) !important;
    border: 1px solid rgba(59, 130, 246, 0.2) !important;
    border-radius: 14px !important;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5) !important;
    backdrop-filter: blur(20px) !important;
}

li[role="option"] {
    color: #e5e7eb !important;
    font-size: 14px !important;
    padding: 12px 16px !important;
    transition: all 0.2s ease !important;
}

li[role="option"]:hover {
    background: rgba(59, 130, 246, 0.2) !important;
    color: #3b82f6 !important;
}

[data-testid="stWidgetLabel"] > div > p {
    color: white !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    margin-bottom: 12px !important;
    letter-spacing: 0.5px !important;
}

/* Columns */
[data-testid="column"] {
    padding: 0 12px !important;
}

/* Button */
[data-testid="stButton"] > button {
    width: 100% !important;
    background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%) !important;
    color: white !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    padding: 18px 40px !important;
    border: none !important;
    border-radius: 16px !important;
    height: auto !important;
    margin-top: 32px !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    box-shadow: 0 20px 40px rgba(59, 130, 246, 0.4) !important;
    cursor: pointer !important;
    text-transform: uppercase;
    font-family: 'Space Mono', monospace !important;
    position: relative;
    overflow: hidden;
}

[data-testid="stButton"] > button::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 0;
    height: 0;
    background: rgba(255, 255, 255, 0.2);
    border-radius: 50%;
    transform: translate(-50%, -50%);
    transition: width 0.6s, height 0.6s;
}

[data-testid="stButton"] > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 30px 60px rgba(59, 130, 246, 0.5) !important;
}

[data-testid="stButton"] > button:active {
    transform: translateY(-1px) !important;
}

/* Result Container */
.result-wrap {
    animation: slideUp 0.8s cubic-bezier(0.16, 1, 0.3, 1);
    margin-top: 60px;
}

@keyframes slideUp {
    from {
        opacity: 0;
        transform: translateY(40px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.result-box {
    border-radius: 28px;
    padding: 60px 50px;
    text-align: center;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(59, 130, 246, 0.2);
    backdrop-filter: blur(20px);
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.05) 0%, rgba(6, 182, 212, 0.03) 100%);
}

.result-box::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.4), transparent);
}

.result-success {
    border-color: rgba(16, 185, 129, 0.3);
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.05) 0%, rgba(6, 182, 212, 0.03) 100%);
}

.result-warning {
    border-color: rgba(239, 68, 68, 0.3);
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.05) 0%, rgba(245, 158, 11, 0.03) 100%);
}

.result-icon {
    font-size: 72px;
    margin-bottom: 20px;
    animation: bounce-smooth 2s ease-in-out infinite;
}

@keyframes bounce-smooth {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-12px); }
}

.result-status {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 12px;
    color: #06b6d4;
}

.result-warning .result-status {
    color: #ef4444;
}

.result-title {
    font-size: 36px;
    font-weight: 800;
    color: white;
    margin-bottom: 20px;
    line-height: 1.2;
}

.result-confidence {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    font-weight: 700;
    padding: 10px 20px;
    border-radius: 24px;
    margin-bottom: 24px;
    background: rgba(59, 130, 246, 0.15);
    color: #3b82f6;
    letter-spacing: 1px;
}

.result-warning .result-confidence {
    background: rgba(239, 68, 68, 0.15);
    color: #ef4444;
}

.result-description {
    font-size: 16px;
    line-height: 1.8;
    color: #9ca3af;
    max-width: 500px;
    margin: 0 auto 40px;
}

.confidence-bar-container {
    max-width: 350px;
    margin: 0 auto;
}

.confidence-label {
    display: flex;
    justify-content: space-between;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: #6b7280;
    margin-bottom: 12px;
    letter-spacing: 1px;
}

.confidence-track {
    height: 8px;
    background: rgba(59, 130, 246, 0.1);
    border-radius: 4px;
    overflow: hidden;
}

.confidence-fill {
    height: 100%;
    background: linear-gradient(90deg, #3b82f6, #06b6d4);
    border-radius: 4px;
    animation: fillBar 1.2s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 0 20px rgba(59, 130, 246, 0.5);
}

.result-warning .confidence-fill {
    background: linear-gradient(90deg, #ef4444, #f59e0b);
    box-shadow: 0 0 20px rgba(239, 68, 68, 0.5);
}

@keyframes fillBar {
    from { width: 0 !important; }
}

/* Disclaimer */
.disclaimer {
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.2);
    border-radius: 16px;
    padding: 20px;
    margin-top: 32px;
    font-size: 13px;
    color: #d4a574;
    line-height: 1.7;
    display: flex;
    gap: 14px;
    align-items: flex-start;
    backdrop-filter: blur(10px);
}

.disclaimer-icon {
    flex-shrink: 0;
    font-size: 18px;
    margin-top: 2px;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 80px;
    padding-top: 40px;
    border-top: 1px solid rgba(59, 130, 246, 0.1);
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    color: #6b7280;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

/* Expander */
details {
    background: rgba(59, 130, 246, 0.05) !important;
    border: 1px solid rgba(59, 130, 246, 0.1) !important;
    border-radius: 14px !important;
    margin-top: 32px !important;
    backdrop-filter: blur(10px) !important;
}

summary {
    color: #9ca3af !important;
    font-size: 12px !important;
    font-family: 'Space Mono', monospace !important;
    padding: 16px !important;
    font-weight: 600 !important;
}

/* Spinner */
[data-testid="stSpinner"] > div {
    border-color: #3b82f6 !important;
}

/* Responsive */
@media (max-width: 768px) {
    .main-wrapper {
        padding: 60px 20px 80px;
    }
    
    .form-section {
        padding: 32px 20px;
    }
    
    .hero h1 {
        font-size: 2.2rem;
    }
    
    .hero-stats {
        gap: 30px;
    }
    
    .result-box {
        padding: 40px 24px;
    }
    
    .result-icon {
        font-size: 56px;
    }
    
    .result-title {
        font-size: 28px;
    }
    
    .hero-stats {
        flex-direction: column;
        gap: 20px;
    }
}

</style>
""", unsafe_allow_html=True)

# ── ANIMATED BACKGROUND HTML ────────────────────────────────────────────────
st.markdown("""
<div class="animated-bg">
    <!-- Gradient Mesh Base -->
    <div class="gradient-mesh"></div>
    
    <!-- Pulsing Auras -->
    <div class="aura aura-1"></div>
    <div class="aura aura-2"></div>
    
    <!-- Organic Blobs -->
    <div class="blob blob-1"></div>
    <div class="blob blob-2"></div>
    <div class="blob blob-3"></div>
    <div class="blob blob-4"></div>
    
    <!-- Light Streaks -->
    <div class="light-streak streak-1"></div>
    <div class="light-streak streak-2"></div>
    <div class="light-streak streak-3"></div>
    
    <!-- Noise -->
    <div class="noise-overlay"></div>
</div>

<div class="main-wrapper">
""", unsafe_allow_html=True)

# ── NAVIGATION BAR ──────────────────────────────────────────────────────────
st.markdown("""
<div class="nav-bar">
    <div class="nav-logo">🧠 MindVault</div>
    <div class="nav-badge">
        <span style="margin-right: 4px;">⚡</span> Cutting Edge AI
    </div>
</div>
""", unsafe_allow_html=True)

# ── HERO SECTION ────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">
        <div class="hero-dot"></div>
        Advanced Mental Health Assessment
    </div>
    <h1>Your Mind <span>Matters</span></h1>
    <p>
        Discover insights about your mental wellbeing through an AI-powered assessment. 
        Answer 14 mindful questions and get personalized, data-driven clarity.
    </p>
    <div class="hero-stats">
        <div class="stat-item">
            <div class="stat-value">72%</div>
            <div class="stat-label">Accuracy</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">&lt;1s</div>
            <div class="stat-label">Analysis</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">100%</div>
            <div class="stat-label">Private</div>
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
<div class="form-section">
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="medium")
with col1:
    gender = st.selectbox("Gender", list(GENDER_MAP.keys()), key="gender_select")
    occupation = st.selectbox("Occupation", list(OCC_MAP.keys()), key="occupation_select")
with col2:
    self_employed = st.selectbox("Are you self-employed?", list(YESNO.keys()), key="self_emp_select")
    family_history = st.selectbox("Family history of mental illness?", list(YESNO.keys()), key="family_select")

st.markdown('</div>', unsafe_allow_html=True)

# ── SECTION 02: STRESS & EMOTIONAL STATE ────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-number">02</div>
    <div class="section-title">Stress & Emotional State</div>
    <div class="section-line"></div>
</div>
<div class="form-section">
""", unsafe_allow_html=True)

col3, col4 = st.columns(2, gap="medium")
with col3:
    growing_stress = st.selectbox("Growing stress lately?", list(YESNO_MAYBE.keys()), key="stress_select")
    mood_swings = st.selectbox("Mood swing intensity?", list(MOOD_MAP.keys()), key="mood_select")
with col4:
    coping_struggles = st.selectbox("Struggling to cope with daily life?", list(YESNO.keys()), key="coping_select")
    changes_habits = st.selectbox("Noticeable changes in habits?", list(YESNO_MAYBE.keys()), key="habits_select")

st.markdown('</div>', unsafe_allow_html=True)

# ── SECTION 03: BEHAVIOR & SOCIAL LIFE ──────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-number">03</div>
    <div class="section-title">Behaviour & Social Life</div>
    <div class="section-line"></div>
</div>
<div class="form-section">
""", unsafe_allow_html=True)

col5, col6 = st.columns(2, gap="medium")
with col5:
    work_interest = st.selectbox("Lost interest in work or hobbies?", list(YESNO_MAYBE.keys()), key="interest_select")
    social_weakness = st.selectbox("Feeling socially withdrawn?", list(YESNO_MAYBE.keys()), key="social_select")
with col6:
    days_indoors = st.selectbox("Days spent indoors (past month)?", list(DAYS_MAP.keys()), key="days_select")

st.markdown('</div>', unsafe_allow_html=True)

# ── SECTION 04: AWARENESS & SUPPORT ─────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <div class="section-number">04</div>
    <div class="section-title">Awareness & Support Access</div>
    <div class="section-line"></div>
</div>
<div class="form-section">
""", unsafe_allow_html=True)

col7, col8 = st.columns(2, gap="medium")
with col7:
    mh_history = st.selectbox("Had mental health issues before?", list(YESNO_MAYBE.keys()), key="history_select")
    mh_interview = st.selectbox("Comfortable discussing MH in interviews?", list(YESNO_MAYBE.keys()), key="interview_select")
with col8:
    care_options = st.selectbox("Access to mental health care?", list(CARE_MAP.keys()), key="care_select")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── PREDICT BUTTON ──────────────────────────────────────────────────────────
predict_clicked = st.button("✨ Reveal Your Assessment", key="predict_button")

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

    with st.spinner("🧠 Analyzing your assessment..."):
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
                <div class="result-wrap">
                    <div class="result-box result-warning">
                        <div class="result-icon">🔴</div>
                        <div class="result-status">Assessment Complete</div>
                        <div class="result-title">Professional Support Recommended</div>
                        <div class="result-confidence">Confidence: {conf_str}</div>
                        <p class="result-description">
                            {explanation or 'Your assessment suggests that professional mental health support would be beneficial.'}
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
                <div class="result-wrap">
                    <div class="result-box result-success">
                        <div class="result-icon">🟢</div>
                        <div class="result-status">Assessment Complete</div>
                        <div class="result-title">Positive Mental State</div>
                        <div class="result-confidence">Confidence: {conf_str}</div>
                        <p class="result-description">
                            {explanation or 'Your assessment indicates a healthy mental state. Keep prioritizing self-care and wellbeing.'}
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
                    <strong>Important:</strong> This assessment is educational only and not a medical diagnosis. 
                    Always consult with a licensed mental health professional for proper guidance.
                </span>
            </div>
            """, unsafe_allow_html=True)

        except requests.exceptions.Timeout:
            st.error("⏱️ Request timeout. Please try again.")
        except requests.exceptions.ConnectionError:
            st.error("🌐 Connection error. Check your network or API configuration.")
        except requests.exceptions.HTTPError as e:
            try:
                detail = response.json()
                error_msg = detail.get("error", str(detail))
            except:
                error_msg = response.text
            st.error(f"❌ API Error: {error_msg}")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    MindVault &nbsp;·&nbsp; Cutting Edge Mental Health Assessment &nbsp;·&nbsp; Educational Purpose
</div>
</div>
""", unsafe_allow_html=True)

# ── DEBUG SECTION ───────────────────────────────────────────────────────────
with st.expander("🔧 Developer Info"):
    st.code(f"API Endpoint: {API_URL}", language="text")
