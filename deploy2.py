import streamlit as st
from google import genai
from google.genai import types

# --- Setup ---
PROJECT_ID = "738928595068"       # your GCP project ID
LOCATION = "us-central1"          # region where you tuned the model
MODEL_ENDPOINT = "projects/738928595068/locations/us-central1/endpoints/7079072574528815104"

# --- Authenticate with API Key ---
if "GOOGLE_CLOUD_API_KEY" not in st.secrets:
    st.error("⚠️ Missing GOOGLE_CLOUD_API_KEY in Streamlit secrets.")
    st.stop()

client = genai.Client(
    vertexai=True,
    api_key=st.secrets["GOOGLE_CLOUD_API_KEY"]
)

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
                # Use a system prompt to guide behavior
                system_prompt = types.Content(
                    role="system",
                    parts=[types.Part.from_text(
                        "You are a friendly and helpful AI assistant. "
                        "Answer all user questions clearly and politely, "
                        "including casual greetings or small talk."
                    )]
                )
                
                user_prompt = types.Content(
                    role="user",
                    parts=[types.Part.from_text(user_input)]
                )
                
                response = client.models.generate_content(
                    model=MODEL_ENDPOINT,
                    contents=[system_prompt, user_prompt],
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
