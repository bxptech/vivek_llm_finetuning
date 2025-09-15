import streamlit as st
from google import genai
from google.genai import types

PROJECT_ID = "738928595068"
LOCATION = "us-central1"

# Your fine-tuned model (for payment-related queries)
FINETUNED_MODEL = "projects/738928595068/locations/us-central1/models/7079072574528815104"

# Use a supported Gemini model (not 1.5 anymore!)
BASE_MODEL = "publishers/google/models/gemini-2.0-flash"

client = genai.Client(vertexai=True, api_key=st.secrets["GOOGLE_CLOUD_API_KEY"])

def is_payment_query(text: str) -> bool:
    keywords = ["payment", "bill", "entry", "invoice", "transaction"]
    return any(kw in text.lower() for kw in keywords)

st.set_page_config(page_title="💡 Hybrid Gemini App", page_icon="✨", layout="centered")
st.title("💡 Hybrid Gemini App")
st.write("Ask me about **payments/bills** (JSON output) or chat casually (normal text).")

user_input = st.text_area("Your prompt", placeholder="Type something...")

if st.button("Generate"):
    if not user_input.strip():
        st.warning("Please enter a prompt first.")
    else:
        with st.spinner("Thinking..."):
            try:
                if is_payment_query(user_input):
                    model = FINETUNED_MODEL
                    st.info("Using **Fine-tuned Payment Model** (JSON output)")
                else:
                    model = BASE_MODEL
                    st.info(f"Using **Base Gemini Model**: {BASE_MODEL}")

                response = client.models.generate_content(
                    model=model,
                    contents=[types.Content(role="user", parts=[types.Part.from_text(user_input)])],
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        max_output_tokens=512
                    )
                )

                st.success("Response:")
                st.write("".join([c.text for c in response.candidates[0].content.parts if c.text]))

            except Exception as e:
                st.error(f"Error: {e}")
