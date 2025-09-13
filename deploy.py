import streamlit as st
from google import genai
from google.genai import types

# --- Setup ---
PROJECT_ID = "738928595068"       # your GCP project ID
LOCATION = "us-central1"          # region where you tuned the model

# Your fine-tuned model (not endpoint, must be a model resource name)
FINETUNED_MODEL = "projects/738928595068/locations/us-central1/models/7079072574528815104"

# --- Authenticate with API Key ---
if "GOOGLE_CLOUD_API_KEY" not in st.secrets:
    st.error("⚠️ Missing GOOGLE_CLOUD_API_KEY in Streamlit secrets.")
    st.stop()

client = genai.Client(
    vertexai=True,
    api_key=st.secrets["GOOGLE_CLOUD_API_KEY"]
)

# --- Streamlit UI ---
st.set_page_config(page_title="💡 Fine-Tuned Gemini Demo", page_icon="✨", layout="centered")

st.title("💡 Fine-Tuned Gemini Demo")
st.write("This app uses **your fine-tuned model** to respond with payment/bill JSON.")

# Input box
user_input = st.text_area("Your prompt", placeholder="Type something like 'Generate payment entry'")

# Button
if st.button("Generate"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            try:
                # Generate response from fine-tuned model
                response = client.models.generate_content(
                    model=FINETUNED_MODEL,
                    contents=[types.Content(role="user", parts=[types.Part.from_text(text=user_input)])],
                    config=types.GenerateContentConfig(
                        temperature=0.0,          # keep it deterministic for JSON
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
