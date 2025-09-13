import streamlit as st
from google import genai
from google.genai import types

# --- Setup ---
PROJECT_ID = "738928595068"
LOCATION = "us-central1"
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
st.set_page_config(page_title="💡 Gemini Adaptive Assistant", page_icon="✨", layout="centered")
st.title("💡 Gemini Adaptive Assistant")
st.write("Ask anything — casual, formal, greetings, or task-specific!")

# Input box
user_input = st.text_area("Your prompt", placeholder="Type something...")

# Button
if st.button("Generate"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            try:
                # Construct prompt with runtime instructions
                prompt_text = (
                    "You are an intelligent AI assistant. "
                    "Analyze the user's question and respond appropriately:\n"
                    "- If the question is casual, respond casually.\n"
                    "- If the question is formal, respond formally.\n"
                    "- If the question is a greeting, respond friendly.\n"
                    "- If the question relates to your fine-tuned task (e.g., transactions), "
                    "respond in the structured format learned from fine-tuning.\n\n"
                    f"User: {user_input}\n"
                    "Assistant:"
                )

                # Create user content
                user_prompt = types.Content(
                    role="user",
                    parts=[types.Part(text=prompt_text)]
                )

                # Generate response
                response = client.models.generate_content(
                    model=MODEL_ENDPOINT,
                    contents=[user_prompt],
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
