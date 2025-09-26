# import streamlit as st
# from google import genai
# from google.genai import types

# # --- Setup ---
# PROJECT_ID = "738928595068"
# LOCATION = "us-central1"
# MODEL_ENDPOINT = "projects/738928595068/locations/us-central1/endpoints/7079072574528815104"

# # --- Authenticate with API Key ---
# if "GOOGLE_CLOUD_API_KEY" not in st.secrets:
#     st.error("⚠️ Missing GOOGLE_CLOUD_API_KEY in Streamlit secrets.")
#     st.stop()

# client = genai.Client(
#     vertexai=True,
#     api_key=st.secrets["GOOGLE_CLOUD_API_KEY"]
# )

# # --- Streamlit UI ---
# st.set_page_config(page_title="My Gemini App", page_icon="✨", layout="centered")

# st.title("💡 Fine-Tuned Gemini Demo")
# st.write("Ask your fine-tuned Gemini model anything:")

# # Input box
# user_input = st.text_area("Your prompt", placeholder="Type something...")

# # Button
# if st.button("Generate"):
#     if user_input.strip():
#         with st.spinner("Thinking..."):
#             try:
#                 # Correct usage of Part.from_text()
#                 system_prompt = types.Content(
#                     role="system",
#                     parts=[types.Part.from_text(
#                         "You are a friendly and helpful AI assistant. "
#                         "Answer all user questions clearly and politely, "
#                         "including casual greetings or small talk."
#                     )]
#                 )
                
#                 user_prompt = types.Content(
#                     role="user",
#                     parts=[types.Part.from_text(user_input)]
#                 )
                
#                 response = client.models.generate_content(
#                     model=MODEL_ENDPOINT,
#                     contents=[system_prompt, user_prompt],
#                     config=types.GenerateContentConfig(
#                         temperature=0.7,
#                         max_output_tokens=512
#                     )
#                 )
                
#                 # Show response
#                 st.success("Response:")
#                 st.write("".join([c.text for c in response.candidates[0].content.parts if c.text]))
            
#             except Exception as e:
#                 st.error(f"Error: {e}")
#     else:
#         st.warning("Please enter a prompt first.") 



# import streamlit as st
# from google import genai
# from google.genai import types
# from google.oauth2 import service_account

# # ----------------------------
# # CONFIGURATION
# # ----------------------------
# PROJECT_ID = "myfinetuning-project"       # your GCP project ID
# LOCATION = "us-central1"                  # region
# FINETUNED_MODEL = (
#     "projects/myfinetuning-project/locations/us-central1/models/vivek_finetuning5"
# )

# # ----------------------------
# # AUTHENTICATION
# # ----------------------------
# # Load service account from Streamlit secrets
# # (put service-account.json contents in .streamlit/secrets.toml as SERVICE_ACCOUNT_KEY)
# if "SERVICE_ACCOUNT_KEY" not in st.secrets:
#     st.error("⚠️ Missing SERVICE_ACCOUNT_KEY in Streamlit secrets.")
#     st.stop()

# credentials = service_account.Credentials.from_service_account_info(
#     st.secrets["SERVICE_ACCOUNT_KEY"]
# )

# client = genai.Client(
#     vertexai=True,
#     project=PROJECT_ID,
#     location=LOCATION,
#     credentials=credentials,
# )

# # ----------------------------
# # STREAMLIT UI
# # ----------------------------
# st.set_page_config(page_title="💡 Fine-Tuned Gemini App", page_icon="✨", layout="centered")
# st.title("💡 Fine-Tuned Gemini App")
# st.write("This app runs on your **Gemini fine-tuned model** in Vertex AI.")

# # Input box
# user_input = st.text_area("Your prompt", placeholder="Type something...")

# # Button
# if st.button("Generate"):
#     if user_input.strip():
#         with st.spinner("Thinking..."):
#             try:
#                 # Send request to fine-tuned model
#                 response = client.models.generate_content(
#                     model=FINETUNED_MODEL,
#                     contents=[types.Content(role="user", parts=[types.Part.from_text(user_input)])],
#                     config=types.GenerateContentConfig(
#                         temperature=0.7,
#                         max_output_tokens=512,
#                     ),
#                 )

#                 # Display output
#                 st.success("Response:")
#                 st.write("".join([c.text for c in response.candidates[0].content.parts if c.text]))

#             except Exception as e:
#                 st.error(f"Error: {e}")
#     else:
#         st.warning("Please enter a prompt first.")

import streamlit as st
import json
import os
from google.oauth2 import service_account
from google import genai
from google.genai import types

st.title("🎯 My Finetuned Gemini Model")

# Load credentials from Streamlit secrets
service_account_info = json.loads(st.secrets["SERVICE_ACCOUNT_JSON"])
credentials = service_account.Credentials.from_service_account_info(service_account_info)

# Initialize client
client = genai.Client(
    vertexai=True,
    project="myfinetuning-project",
    location="us-central1",
    credentials=credentials,
)

# Select your model
MODEL_NAME = "projects/myfinetuning-project/locations/us-central1/models/vivek_finetuning6"

# User input
user_input = st.text_input("💬 Enter your message:")

if st.button("Generate Response"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_input)]
            )
        ]

        generate_content_config = types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.95,
            max_output_tokens=1024,
            safety_settings=[],
        )

        response_text = ""
        for chunk in client.models.generate_content_stream(
            model=MODEL_NAME,
            contents=contents,
            config=generate_content_config,
        ):
            response_text += chunk.text or ""

        st.success("✅ Response from model:")
        st.write(response_text)
