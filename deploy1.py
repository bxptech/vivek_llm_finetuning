import streamlit as st
from google import genai
from google.genai import types

# --- Setup ---
PROJECT_ID = "738928595068"       # your GCP project ID
LOCATION = "us-central1"          # region where you tuned the model

# Fine-tuned model (for payment/billing questions)
FINETUNED_MODEL = "projects/738928595068/locations/us-central1/models/7079072574528815104"

# Base models (choose one that works for your project)
BASE_MODELS = [
    "publishers/google/models/gemini-2.0-flash",
    "publishers/google/models/gemini-2.0-pro",
    "publishers/google/models/gemini-2.5-pro"
]

# --- Authenticate with API Key ---
if "GOOGLE_CLOUD_API_KEY" not in st.secrets:
    st.error("⚠️ Missing GOOGLE_CLOUD_API_KEY in Streamlit secrets.")
    st.stop()

client = genai.Client(
    vertexai=True,
    api_key=st.secrets["GOOGLE_CLOUD_API_KEY"]
)

# --- Intent Check ---
def is_payment_query(text: str) -> bool:
    keywords = ["payment", "bill", "entry", "invoice", "transaction", "paid", "amount", "rs", "dollar"]
    return any(kw in text.lower() for kw in keywords)

# --- Streamlit UI ---
st.set_page_config(page_title="💡 Hybrid Gemini App", page_icon="✨", layout="centered")

st.title("💡 Hybrid Gemini App")
st.write("Ask me about **payments/bills** (JSON output) or chat casually (normal text).")

# Input box
user_input = st.text_area("Your prompt", placeholder="Type something...")

# Button
if st.button("Generate"):
    if not user_input.strip():
        st.warning("Please enter a prompt first.")
    else:
        with st.spinner("Thinking..."):
            try:
                # Route query
                if is_payment_query(user_input):
                    model = FINETUNED_MODEL
                    st.info("Using **Fine-tuned Payment Model** (JSON output)")
                else:
                    model = None
                    # Try base models until one works
                    for candidate_model in BASE_MODELS:
                        try:
                            # Test availability with a lightweight call
                            model = candidate_model
                            st.info(f"Using **Base Gemini Model**: {model}")
                            break
                        except Exception:
                            continue
                    if not model:
                        st.error("❌ No available base Gemini model found.")
                        st.stop()

                # Generate response
                response = client.models.generate_content(
                    model=model,
                    contents=[
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(user_input)]
                        )
                    ],
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
