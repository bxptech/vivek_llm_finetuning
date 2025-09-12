import streamlit as st
import os
import json
from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import GenerativeModel

# --- Setup ---
PROJECT_ID = "738928595068"       # your GCP project ID
LOCATION = "us-central1"          # region where you tuned the model
TUNED_MODEL_ID = "1234567890123456789"  # replace with your tuned model ID

# --- Authenticate using Streamlit Cloud Secrets ---
if "GCP_CREDENTIALS" not in st.secrets:
    st.error("⚠️ Missing GCP_CREDENTIALS in Streamlit secrets.")
    st.stop()

creds_dict = json.loads(st.secrets["GCP_CREDENTIALS"])
credentials = service_account.Credentials.from_service_account_info(creds_dict)

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)

# Load tuned Gemini model
model = GenerativeModel(f"projects/{PROJECT_ID}/locations/{LOCATION}/models/{TUNED_MODEL_ID}")

# --- Streamlit UI ---
st.set_page_config(page_title="My Gemini App", page_icon="✨", layout="centered")

st.title("💡 Fine-Tuned Gemini Demo")
st.write("Ask your fine-tuned Gemini model anything:")

# Input box
user_input = st.text_area("Your prompt", placeholder="Type something...")

# Button
if st.button("Generate"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            try:
                response = model.generate_content(user_input)
                st.success("Response:")
                st.write(response.text)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a prompt first.")
