import streamlit as st
from google import genai
from google.genai import types
from google.oauth2 import service_account

# ----------------------------
# CONFIGURATION
# ----------------------------
PROJECT_ID = "myfinetuning-project"       # your GCP project ID
LOCATION = "us-central1"                  # region
FINETUNED_MODEL = (
    "projects/myfinetuning-project/locations/us-central1/models/vivek_finetuning5"
)

# ----------------------------
# AUTHENTICATION
# ----------------------------
# Load service account from Streamlit secrets
# (put service-account.json contents in .streamlit/secrets.toml as SERVICE_ACCOUNT_KEY)
if "SERVICE_ACCOUNT_KEY" not in st.secrets:
    st.error("⚠️ Missing SERVICE_ACCOUNT_KEY in Streamlit secrets.")
    st.stop()

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["SERVICE_ACCOUNT_KEY"]
)

client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
    credentials=credentials,
)

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.set_page_config(page_title="💡 Fine-Tuned Gemini App", page_icon="✨", layout="centered")
st.title("💡 Fine-Tuned Gemini App")
st.write("This app runs on your **Gemini fine-tuned model** in Vertex AI.")

# Input box
user_input = st.text_area("Your prompt", placeholder="Type something...")

# Button
if st.button("Generate"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            try:
                # Send request to fine-tuned model
                response = client.models.generate_content(
                    model=FINETUNED_MODEL,
                    contents=[types.Content(role="user", parts=[types.Part.from_text(user_input)])],
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        max_output_tokens=512,
                    ),
                )

                # Display output
                st.success("Response:")
                st.write("".join([c.text for c in response.candidates[0].content.parts if c.text]))

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a prompt first.")
