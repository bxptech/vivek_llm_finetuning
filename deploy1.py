import streamlit as st
from google import genai
from google.genai import types

# --- Setup ---
PROJECT_ID = "738928595068"
LOCATION = "us-central1"
TUNED_MODEL_ENDPOINT = "projects/738928595068/locations/us-central1/endpoints/7079072574528815104"
BASE_MODEL = "gemini-2.5-flash"   # ✅ supported latest model for general chat

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

st.title("💡 Gemini + Fine-Tuned Model")
st.write("Ask me anything. I'll choose the right brain:")

# Input box
user_input = st.text_area("Your prompt", placeholder="Type something...")

# --- Simple intent detection ---
def is_transaction_query(text: str) -> bool:
    """Decide if query looks like a transaction request."""
    keywords = ["pay", "paid", "send", "transfer", "deposit", "withdraw", "cash", "rs", "$", "amount"]
    return any(kw in text.lower() for kw in keywords)

# --- Button ---
if st.button("Generate"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            try:
                if is_transaction_query(user_input):
                    # Use fine-tuned model for structured JSON
                    response = client.models.generate_content(
                        model=TUNED_MODEL_ENDPOINT,
                        contents=[types.Content(role="user", parts=[types.Part.from_text(text=user_input)])],
                        config=types.GenerateContentConfig(
                            temperature=0.3,  # lower for consistency
                            max_output_tokens=512
                        )
                    )
                    output = "".join([c.text for c in response.candidates[0].content.parts if c.text])
                    st.success("Response (from Fine-Tuned Model):")
                    st.code(output, language="json")
                else:
                    # Use base Gemini model for general chat
                    response = client.models.generate_content(
                        model=BASE_MODEL,
                        contents=[types.Content(role="user", parts=[types.Part.from_text(text=user_input)])],
                        config=types.GenerateContentConfig(
                            temperature=0.7,
                            max_output_tokens=512
                        )
                    )
                    output = "".join([c.text for c in response.candidates[0].content.parts if c.text])
                    st.success("Response (from Base Model):")
                    st.write(output)

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a prompt first.")
