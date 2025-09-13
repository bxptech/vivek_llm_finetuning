import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel
from google.oauth2 import service_account
import json

# --- Setup ---
PROJECT_ID = "738928595068"
LOCATION = "us-central1"
TUNED_MODEL_ID = "7079072574528815104"   # should be a MODEL id, not endpoint

# --- Authenticate ---
if "GCP_CREDENTIALS" not in st.secrets:
    st.error("⚠️ Missing GCP_CREDENTIALS in Streamlit secrets.")
    st.stop()

creds_dict = json.loads(st.secrets["GCP_CREDENTIALS"])
credentials = service_account.Credentials.from_service_account_info(creds_dict)

vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)

# Load fine-tuned Gemini model
model = GenerativeModel(f"projects/{PROJECT_ID}/locations/{LOCATION}/models/{TUNED_MODEL_ID}")

# --- Streamlit UI ---
st.set_page_config(page_title="💡 Fine-Tuned Gemini Demo", page_icon="✨")
st.title("💡 Fine-Tuned Gemini Demo")

user_input = st.text_area("Your prompt", placeholder="Type something...")

if st.button("Generate"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            try:
                response = model.generate_content(user_input)
                st.success("Response:")
                st.write(response.text)
            except Exception as e:
                st.error(f"❌ Error: {e}")
