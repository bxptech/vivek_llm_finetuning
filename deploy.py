import streamlit as st
from google import genai
from google.genai import types

# --- Setup ---
PROJECT_ID = "738928595068"       # your GCP project ID
LOCATION = "us-central1"          # region where you tuned the model
TUNED_MODEL = "projects/738928595068/locations/us-central1/endpoints/7079072574528815104"
BASE_MODEL = "gemini-1.5-pro"     # base conversational model

# --- Authenticate with API Key ---
if "GOOGLE_CLOUD_API_KEY" not in st.secrets:
    st.error("⚠️ Missing GOOGLE_CLOUD_API_KEY in Streamlit secrets.")
    st.stop()

client = genai.Client(
    vertexai=True,
    api_key=st.secrets["GOOGLE_CLOUD_API_KEY"]
)

# --- Router function ---
def is_payment_query(text: str) -> bool:
    keywords = ["payment", "bill", "entry"]
    return any(kw in text.lower() for kw in keywords)

def call_model(model_name: str, user_input: str):
    response = client.models.generate_content(
        model=model_name,
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=user_input)])],
        config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=512
        )
    )
    return "".join([c.text for c in response.candidates[0].content.parts if c.text])

# --- Streamlit UI ---
st.set_page_config(page_title="My Gemini App", page_icon="✨", layout="centered")

st.title("💡 Fine-Tuned Gemini Demo (Hybrid)")
st.write("This app routes your query: \n- **Payment/Bill/Entry → Fine-tuned model (JSON)** \n- **Everything else → Base Gemini model (chat)**")

# Input box
user_input = st.text_area("Your prompt", placeholder="Type something...")

# Button
if st.button("Generate"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            try:
                # Route based on intent
                if is_payment_query(user_input):
                    model_to_use = TUNED_MODEL
                else:
                    model_to_use = BASE_MODEL

                response_text = call_model(model_to_use, user_input)

                # Show response
                st.success("Response:")
                st.write(response_text)

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a prompt first.")
