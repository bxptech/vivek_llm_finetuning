import streamlit as st
from google import genai
from google.genai import types

# --- Setup ---
PROJECT_ID = "738928595068"
LOCATION = "us-central1"

# Fine-tuned model (task-specific)
FINETUNED_MODEL = "projects/738928595068/locations/us-central1/endpoints/7079072574528815104"

# General-purpose Gemini model (not fine-tuned)
GENERAL_MODEL = "text-bison-001"  # replace with your deployed general-purpose model if needed

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

# Simple keyword-based detection of question type
def is_casual_or_greeting(text):
    keywords = ["hi", "hello", "hey", "how are you", "good morning", "good night", "what's up"]
    return any(k in text.lower() for k in keywords)

def is_task_specific(text):
    # You can improve this with regex or keywords specific to your fine-tuned task
    task_keywords = ["transaction", "amount", "transfer", "cash", "payment"]
    return any(k in text.lower() for k in task_keywords)

# Button
if st.button("Generate"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            try:
                # Determine which model to use
                if is_task_specific(user_input):
                    model_to_use = FINETUNED_MODEL
                    prompt_text = user_input  # fine-tuned model expects task prompt directly
                else:
                    model_to_use = GENERAL_MODEL
                    # General model instructions for casual/formal/greetings
                    prompt_text = (
                        "You are a friendly and intelligent AI assistant. "
                        "Answer casual questions casually, formal questions formally, "
                        "and greetings appropriately.\n\n"
                        f"User: {user_input}\n"
                        "Assistant:"
                    )

                # Prepare content
                user_prompt = types.Content(
                    role="user",
                    parts=[types.Part(text=prompt_text)]
                )

                # Generate response
                response = client.models.generate_content(
                    model=model_to_use,
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
