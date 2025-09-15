import streamlit as st
from google import genai
from google.genai import types

# --- Setup ---
PROJECT_ID = "738928595068"
LOCATION = "us-central1"
TUNED_MODEL_ENDPOINT = "projects/738928595068/locations/us-central1/endpoints/7079072574528815104"
BASE_MODEL = "gemini-2.5-flash"   # ✅ use supported stable model

# --- Authenticate ---
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

user_input = st.text_area("Your prompt", placeholder="Type something...")

# --- Intent Detection ---
def is_transaction_query(text: str) -> bool:
    keywords = ["pay", "send", "transfer", "deposit", "withdraw", "account", "cash"]
    return any(kw in text.lower() for kw in keywords)

# --- Generate ---
if st.button("Generate"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            try:
                if is_transaction_query(user_input):
                    # Use fine-tuned model for transactions
                    response = client.models.generate_content(
                        model=TUNED_MODEL_ENDPOINT,
                        contents=[types.Content(role="user", parts=[types.Part.from_text(text=user_input)])],
                        config=types.GenerateContentConfig(
                            temperature=0.3,
                            max_output_tokens=512
                        )
                    )
                else:
                    # Use base Gemini 2.5 model for chat
                    response = client.models.generate_content(
                        model=BASE_MODEL,
                        contents=[types.Content(role="user", parts=[types.Part.from_text(text=user_input)])],
                        config=types.GenerateContentConfig(
                            temperature=0.7,
                            max_output_tokens=512
                        )
                    )

                # Show response
                st.success("Response:")
                st.write("".join([
                    c.text for c in response.candidates[0].content.parts if c.text
                ]))
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a prompt first.")
