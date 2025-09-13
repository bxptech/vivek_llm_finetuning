import streamlit as st
import json
from google.oauth2 import service_account
from google.cloud import aiplatform_v1

# --- Setup ---
PROJECT_ID = "738928595068"
LOCATION = "us-central1"
ENDPOINT_ID = "YOUR_ENDPOINT_ID"   # after you deploy the tuned model

# --- Authenticate ---
if "GCP_CREDENTIALS" not in st.secrets:
    st.error("⚠️ Missing GCP_CREDENTIALS in Streamlit secrets.")
    st.stop()

creds_dict = json.loads(st.secrets["GCP_CREDENTIALS"])
credentials = service_account.Credentials.from_service_account_info(creds_dict)

client = aiplatform_v1.PredictionServiceClient(credentials=credentials)

endpoint = client.endpoint_path(project=PROJECT_ID, location=LOCATION, endpoint=ENDPOINT_ID)

# --- Streamlit UI ---
st.set_page_config(page_title="Fine-Tuned Gemini", page_icon="✨", layout="centered")

st.title("💡 Fine-Tuned Gemini Demo")
st.write("Ask your fine-tuned Gemini model anything:")

user_input = st.text_area("Your prompt", placeholder="Type something...")

if st.button("Generate"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            try:
                # Build the request payload (your tuned model may need JSON format)
                instance = {"prompt": user_input}

                response = client.predict(
                    endpoint=endpoint,
                    instances=[instance],
                )

                st.success("Response:")
                st.write(response.predictions)

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a prompt first.")
