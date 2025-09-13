import streamlit as st
import json
from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import GenerativeModel

# --- Setup ---
PROJECT_ID = st.secrets["general"]["PROJECT_ID"]
LOCATION = st.secrets["general"]["LOCATION"]
TUNED_MODEL_ID = st.secrets["general"]["TUNED_MODEL_ID"]

# --- Authenticate ---
creds_dict = dict(st.secrets["GCP_CREDENTIALS"])
credentials = service_account.Credentials.from_service_account_info(creds_dict)

vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)

# Load fine-tuned Gemini model directly
model = GenerativeModel(f"projects/{PROJECT_ID}/locations/{LOCATION}/models/{TUNED_MODEL_ID}")

# --- Streamlit UI ---
st.set_page_config(page_title="💡 Fine-tuned Gemini Demo", page_icon="✨", layout="centered")

st.title("💡 Fine-tuned Gemini Demo")
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
