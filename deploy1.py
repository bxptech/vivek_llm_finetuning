import streamlit as st
from google import genai
from google.genai import types
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
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

# --- Build semantic retriever for in-domain detection ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Example domain docs (use your fine-tuned Q&A pairs instead)
documents = [
    Document(page_content="Transaction failures happen when insufficient balance exists."),
    Document(page_content="You can transfer money using the app."),
    Document(page_content="Payment receipts are generated instantly."),
    Document(page_content="Cash withdrawal is possible at ATMs."),
]

# Split + store
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
docs = splitter.split_documents(documents)
vectorstore = FAISS.from_documents(docs, embeddings)

# --- Helper function ---
def is_task_specific(question: str, threshold: float = 0.75) -> bool:
    """Check if the question is close to fine-tuned domain"""
    related = vectorstore.similarity_search(question, k=1)
    if not related:
        return False
    q_emb = embeddings.embed_query(question)
    d_emb = embeddings.embed_query(related[0].page_content)
    similarity = sum(qe * de for qe, de in zip(q_emb, d_emb)) / (
        (sum(q**2 for q in q_emb) ** 0.5) * (sum(d**2 for d in d_emb) ** 0.5)
    )
    return similarity >= threshold

# --- Streamlit UI ---
st.set_page_config(page_title="💡 Gemini Adaptive Assistant", page_icon="✨", layout="centered")
st.title("💡 Gemini Adaptive Assistant")
st.write("Ask anything — task-specific Qs go to finetuned Gemini, others to general Gemini!")

user_input = st.text_area("Your prompt", placeholder="Type something...")

if st.button("Generate"):
    if user_input.strip():
        with st.spinner("Thinking..."):
            try:
                if is_task_specific(user_input):
                    model_to_use = FINETUNED_MODEL
                    prompt_text = user_input
                else:
                    model_to_use = GENERAL_MODEL
                    prompt_text = (
                        "You are a helpful, friendly AI assistant. "
                        "If this question is outside the fine-tuned domain, answer normally:\n\n"
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
