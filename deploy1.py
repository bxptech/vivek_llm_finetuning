import streamlit as st
from google import genai
from google.genai import types

# --- Setup ---
PROJECT_ID = "738928595068"
LOCATION = "us-central1"

# Fine-tuned model endpoint
FINETUNED_MODEL = "projects/738928595068/locations/us-central1/endpoints/7079072574528815104"

# General-purpose Gemini model
GENERAL_MODEL = "gemini-2.0-flash"

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
st.write("Ask anything — task-specific questions go to your **finetuned Gemini**, others to general Gemini!")

user_input = st.text_area("Your prompt", placeholder="Type something...")

# --- Detection: Ask general Gemini if question is in-domain ---
def is_in_domain(question: str) -> bool:
    """Use general Gemini model to classify if question fits the finetuned domain"""
    classification_prompt = f"""
    You are a classifier. 
    Decide if the following user question is related to **finance, transactions, payments, or money transfers** (the fine-tuned model domain).
    Answer only "YES" or "NO".

    User question: {question}
    """

    user_prompt = types.Content(
        role="user",
        parts=[types.Part(text=classification_prompt)]
    )

    response = client.models.generate_content(
        model=GENERAL_MODEL,
        contents=[user_prompt],
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=5
        )
    )

    answer = "".join([c.text for c in response.candidates[0].content.parts if c.text]).strip().upper()
    return answer.startswith("Y")

# --- Generate Response ---
if st.button("Generate"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            try:
                if is_in_domain(user_input):
                    # Use fine-tuned model
                    model_to_use = FINETUNED_MODEL
                    prompt_text = user_input
                else:
                    # Use general model
                    model_to_use = GENERAL_MODEL
                    prompt_text = (
                        "You are a helpful, friendly AI assistant. "
                        "Answer normally since this is outside the fine-tuned scope.\n\n"
                        f"User: {user_input}\nAssistant:"
                    )

                user_prompt = types.Content(
                    role="user",
                    parts=[types.Part(text=prompt_text)]
                )

                response = client.models.generate_content(
                    model=model_to_use,
                    contents=[user_prompt],
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        max_output_tokens=512
                    )
                )

                st.success("Response:")
                st.write("".join([c.text for c in response.candidates[0].content.parts if c.text]))

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a prompt first.")
