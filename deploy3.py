import streamlit as st
from google import genai
from google.genai import types
from google.oauth2 import service_account
# --- Setup ---
PROJECT_ID = "myfinetuning-project"   # project_id from your service account
LOCATION = "us-central1"              # region where you tuned the model
MODEL_ENDPOINT = "projects/myfinetuning-project/locations/us-central1/endpoints/7079072574528815104"

# --- Authenticate with Service Account ---
try:
    # Load service account from Streamlit secrets
    service_account_info = st.secrets["service_account"]
    credentials = service_account.Credentials.from_service_account_info(service_account_info)

    client = genai.Client(
        vertexai=True,
        credentials=credentials
    )
except Exception as e:
    st.error(f"⚠️ Authentication error: {e}")
    st.stop()

# --- Streamlit UI ---
st.set_page_config(page_title="My Gemini App", page_icon="✨", layout="centered")

st.title("💡 Fine-Tuned Gemini Demo")
st.write("Ask your fine-tuned Gemini model anything:")

# Input box
user_input = st.text_area("Your prompt", placeholder="Type something...")

# Predefined common question responses
common_responses = {
    "hello": "Hi there! How’s your day going?",
    "hi": "Hello! Hope you're doing well!",
    "hey": "Hey! How can I help you today?",
    "how are you?": "I'm doing great, thanks for asking! How about you?",
    "what is your name?": "I’m your friendly Gemini assistant!",
    "good morning": "Good morning! Hope you have a fantastic day!",
    "good night": "Good night! Sleep well!"
}

# Button
if st.button("Generate"):
    if user_input.strip():
        # Check for common questions first (case insensitive)
        lower_input = user_input.strip().lower()
        matched = False
        for key in common_responses:
            if key in lower_input:
                st.success("Response:")
                st.write(common_responses[key])
                matched = True
                break

        # If not a common question, call the fine-tuned model
        if not matched:
            with st.spinner("Thinking..."):
                try:
                    response = client.models.generate_content(
                        model=MODEL_ENDPOINT,
                        contents=[types.Content(role="user", parts=[types.Part.from_text(text=user_input)])],
                        config=types.GenerateContentConfig(
                            temperature=0.7,
                            max_output_tokens=512
                        )
                    )
                    # Show response
                    st.success("Response:")
                    st.write("".join([c.text for c in response.candidates[0].content.parts if c.text]))
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.warning("Please enter a prompt first.")
