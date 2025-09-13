import streamlit as st
import json
from google.oauth2 import service_account
from google.cloud import aiplatform

# --- Setup ---
PROJECT_ID = "738928595068"       # your GCP project ID
LOCATION = "us-central1"          # region
ENDPOINT_ID = "7079072574528815104"  # your deployed endpoint ID

# --- Authenticate with Service Account ---
if "GCP_CREDENTIALS" not in st.secrets:
    st.error("⚠️ Missing GCP_CREDENTIALS in Streamlit secrets.")
    st.stop()

creds_dict = json.loads(st.secrets["GCP_CREDENTIALS"])
credentials = service_account.Credentials.from_service_account_info(creds_dict)

# Initialize Prediction client
aiplatform.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
endpoint = aiplatform.Endpoint(endpoint_name=f"projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}")

# --- Streamlit UI ---
st.set_page_config(page_title="💡 Fine-tuned Gemini Demo", page_icon="✨", layout="centered")
st.title("💡 Fine-tuned Gemini Demo")
st.write("This app uses **your fine-tuned model** deployed on Vertex AI endpoint.")

# Input box
user_input = st.text_area("Your prompt", placeholder="Type something like 'Generate payment entry'")

# Button
if st.button("Generate"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            try:
                # Call endpoint for prediction
                response = endpoint.predict(instances=[{"prompt": user_input}])

                # Show response
                st.success("Response:")
                st.write(response.predictions)

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a prompt first.")
