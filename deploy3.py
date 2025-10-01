import streamlit as st
from google import genai
from google.genai import types
from google.oauth2 import service_account

# --- Config ---
PROJECT_ID = "myfinetuning-project"   # From your service account
LOCATION = "us-central1"              # Region of your tuned model
MODEL_ENDPOINT = "projects/myfinetuning-project/locations/us-central1/endpoints/7079072574528815104"

# --- Authenticate with Service Account ---
try:
    # Load service account from Streamlit secrets
    service_account_info = st.secrets["service_account"]
    credentials = service_account.Credentials.from_service_account_info(service_account_info)

    # Create a GenAI client with Vertex AI support
    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
        credentials=credentials
    )

except Exception as e:
    st.error(f"⚠️ Authentication error: {e}")
    st.stop()

# --- Streamlit UI Setup ---
st.set_page_config(page_title="✨ Gemini Fine-Tuned Demo", page_icon="🤖", layout="centered")

st.title("💡 Fine-Tuned Gemini Demo")
st.write("Ask your fine-tuned Gemini model anything below:")

# User input
user_input = st.text_area("Your prompt", placeholder="Type something...")

# Predefined friendly responses
common_responses = {
    "hello": "Hi there! How’s your day going?",
    "hi": "Hello! Hope you're doing well!",
    "hey": "Hey! How can I help you today?",
    "how are you?": "I'm doing great, thanks for asking! How about you?",
    "what is your name?": "I’m your friendly Gemini assistant!",
    "good morning": "Good morning! Hope you have a fantastic day!",
    "good night": "Good night! Sleep well!"
}

# --- Generate Response ---
if st.button("✨ Generate Response"):
    if user_input.strip():
        lower_input = user_input.strip().lower()

        # Check for predefined responses
        if lower_input in common_responses:
            st.success("Response:")
            st.write(common_responses[lower_input])
        else:
            with st.spinner("Thinking... 🤔"):
                try:
                    response = client.models.generate_content(
                        model=MODEL_ENDPOINT,
                        contents=[
                            types.Content(
                                role="user",
                                parts=[types.Part.from_text(text=user_input)]
                            )
                        ],
                        config=types.GenerateContentConfig(
                            temperature=0.7,
                            max_output_tokens=512
                        )
                    )

                    # Extract response safely
                    if response and response.candidates:
                        text_out = "".join(
                            part.text for part in response.candidates[0].content.parts if part.text
                        )
                        st.success("Response:")
                        st.write(text_out)
                    else:
                        st.warning("⚠️ No response generated. Try another input.")

                except Exception as e:
                    st.error(f"❌ Error while generating: {e}")
    else:
        st.warning("⚠️ Please enter a prompt first.")
