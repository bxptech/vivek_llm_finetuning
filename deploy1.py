import streamlit as st
from google import genai
from google.genai import types

# --- Setup ---
PROJECT_ID = "738928595068"       # your GCP project ID
LOCATION = "us-central1"          # region where you tuned the model

# IMPORTANT: Must be a MODEL resource name, not an endpoint
# Go to Vertex AI → Models in the console to confirm this ID
FINETUNED_MODEL = f"projects/{PROJECT_ID}/locations/{LOCATION}/models/7079072574528815104"

# --- Authenticate with API Key ---
if "GOOGLE_CLOUD_API_KEY" not in st.secrets:
    st.error("⚠️ Missing GOOGLE_CLOUD_API_KEY in Streamlit secrets.")
    st.stop()

client = genai.Client(
    vertexai=True,  # required when using Vertex AI models
    api_key=st.secrets["GOOGLE_CLOUD_API_KEY"]
)

# --- Streamlit UI ---
st.set_page_config(page_title="💡 Fine-Tuned Gemini Demo", page_icon="✨", layout="centered")

st.title("💡 Fine-Tuned Gemini Demo")
st.write("This app connects to your **fine-tuned Gemini model** on Vertex AI.")

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
                    contents=[
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=user_input)]
                        )
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0.0,          # deterministic JSON
                        max_output_tokens=512
                    )
                )

                # Extract response text
                if response.candidates and response.candidates[0].content.parts:
                    output_text = "".join(
                        [p.text for p in response.candidates[0].content.parts if hasattr(p, "text")]
                    )
                else:
                    output_text = "(No response text received.)"

                # Show response
                st.success("Response:")
                st.code(output_text, language="json")

            except Exception as e:
                st.error(f"❌ Error: {e}")
    else:
        st.warning("⚠️ Please enter a prompt first.")
