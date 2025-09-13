import streamlit as st
from google import genai
from google.genai import types

# --- Setup ---
PROJECT_ID = "738928595068"
LOCATION = "us-central1"

# ⚠️ Replace with your fine-tuned MODEL ID (not endpoint!)
FINETUNED_MODEL = "projects/738928595068/locations/us-central1/models/7079072574528815104"

# --- Auth ---
if "GOOGLE_CLOUD_API_KEY" not in st.secrets:
    st.error("⚠️ Missing GOOGLE_CLOUD_API_KEY in Streamlit secrets.")
    st.stop()

client = genai.Client(vertexai=True, api_key=st.secrets["GOOGLE_CLOUD_API_KEY"])

# --- UI ---
st.title("💡 Fine-tuned Gemini Demo")

user_input = st.text_area("Your prompt", placeholder="Type 'Generate payment entry'")

if st.button("Generate"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            try:
                response = client.models.generate_content(
                    model=FINETUNED_MODEL,
                    contents=[types.Content(role="user", parts=[types.Part.from_text(text=user_input)])],
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=512
                    )
                )
                st.success("Response:")
                st.write("".join([c.text for c in response.candidates[0].content.parts if c.text]))
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a prompt first.")
