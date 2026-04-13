import streamlit as st
import requests

st.title("Mental Health Predictor")

# Get user inputs...
# Then call Railway API:

response = requests.post(
    "https://mentalhealthcrisispredictor-production.up.railway.app/predict",
    json={...}
)

result = response.json()
st.write(result)
