import streamlit as st
from google import genai
from google.genai import types

# --- Setup ---
PROJECT_ID = "738928595068"
LOCATION = "us-central1"

# Fine-tuned payment model (full path is required)
FINETUNED_MODEL = "projects/738928595068/locations/us-central1/models/7079072574528815104"

# Base Gemini model (short name for Vertex AI)
BASE_MODEL = "gemini-1.5-pro"

# --- Authenticate ---
if "GOOGLE_CLOUD_API_KEY" not in st.secrets:
    st.error("⚠️ Missing GOOGLE_CLOUD_API_KEY in Streamlit secrets.")
    st.stop()

client = genai.Client(
    vertexai=True,
    api_key=st.secrets["GOOGLE_CLOUD_API_KEY"]
)

# --- Intent Classification ---
def is_relevant_to_payment(query: str) -> bool:
    """Ask the base model if the query is about payment/bill entry."""
    try:
        resp = client.models.generate_content(
            model=BASE_MODEL,
            contents=[types.Content(
                role="user",
                parts=[types.Part.from_text(
                    text=f"Classify the following query: '{query}'. "
                         "Answer only YES if it is about creating a payment/bill/transaction entry "
                         "(structured JSON output). Otherwise answer NO."
                )]
            )],
            config=types.GenerateContentConfig(
                temperature=0.0,  # deterministic classification
                max_output_tokens=5
            )
        )

        answer = "".join(
            [c.text for c in resp.candidates[0].content.parts if hasattr(c, "text")]
        ).strip().upper()

        return answer.startswith("Y")
    except Exception as e:
        st.warning(f"Classification failed: {e}")
        return False

# --- Streamlit UI ---
st.set_page_config(page_title="💡 Hybrid Gemini App", page_icon="✨", layout="centered")
st.title("💡 Hybrid Gemini App")
st.write("👉 Ask about **payments/bills** (JSON output from fine-tuned model) or chat casually (base Gemini).")

# Input box
user_input = st.text_area("Your prompt", placeholder="Type something in English or Telugu...")

# Button
if st.button("Generate"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            try:
                # Decide which model to use
                if is_relevant_to_payment(user_input):
                    model = FINETUNED_MODEL
                    st.info("✅ Using Fine-tuned Payment Model (JSON output)")
                else:
                    model = BASE_MODEL
                    st.info("💬 Using Base Gemini Model (conversational output)")

                # Generate response
                response = client.models.generate_content(
                    model=model,
                    contents=[types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=user_input)]
                    )],
                    config=types.GenerateContentConfig(
                        temperature=0.3 if model == FINETUNED_MODEL else 0.7,
                        max_output_tokens=512
                    )
                )

                # Extract text parts safely
                output_text = "".join(
                    [c.text for c in response.candidates[0].content.parts if hasattr(c, "text")]
                )

                # Show response
                st.success("Response:")
                if model == FINETUNED_MODEL:
                    if output_text.strip().startswith("{"):
                        st.json(output_text)  # pretty JSON
                    else:
                        st.write(output_text)
                else:
                    st.write(output_text)

            except Exception as e:
                st.error(f"❌ Error: {e}")
    else:
        st.warning("Please enter a prompt first.")
