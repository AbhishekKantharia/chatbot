import os
import uuid
import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema.runnable import RunnableLambda

# ------------------ 🏗️ Streamlit Page Setup ------------------
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 AI Chatbot with Context & Multi-User Support")

# ------------------ 🔑 API Key Handling ------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = st.sidebar.text_input("Enter Google AI API Key:", type="password")
    if not GOOGLE_API_KEY:
        st.sidebar.error("API key required!")
        st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# ------------------ 🆔 Multi-User Support ------------------
if "user_id" not in st.session_state:
    st.session_state["user_id"] = str(uuid.uuid4())

user_id = st.session_state["user_id"]

if "messages" not in st.session_state:
    st.session_state.messages = {}

if user_id not in st.session_state.messages:
    st.session_state.messages[user_id] = []

if "memory" not in st.session_state:
    st.session_state.memory = {}

if user_id not in st.session_state.memory:
    st.session_state.memory[user_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

if "context_docs" not in st.session_state:
    st.session_state.context_docs = {}

if user_id not in st.session_state.context_docs:
    st.session_state.context_docs[user_id] = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = {}

# ------------------ 🎭 Chatbot Theme Selection ------------------
st.sidebar.subheader("🤖 Choose Chatbot Mode")
chatbot_mode = st.sidebar.radio(
    "Select a theme",
    ["General AI", "Customer Support", "HR Assistant", "Finance Assistant"]
)

mode_prompts = {
    "General AI": "You are a helpful AI assistant.",
    "Customer Support": "You are a customer service agent. Help users with inquiries professionally.",
    "HR Assistant": "You are an HR assistant, helping with policies, onboarding, and HR queries.",
    "Finance Assistant": "You are a finance expert. Help users with financial queries, budgeting, and reports."
}

system_prompt = mode_prompts[chatbot_mode]

# Initialize LLM with theme-based behavior
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7, system=system_prompt)

# ------------------ 📂 File Upload for Document-Based RAG ------------------
st.sidebar.subheader("📂 Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs, DOCX, or TXT files",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True,
)

def process_file(file):
    """Extract text from uploaded files."""
    if file.type == "application/pdf":
        pdf_reader = PdfReader(file)
        text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    elif file.type == "text/plain":
        text = file.getvalue().decode("utf-8")
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = file.getvalue().decode("utf-8")  # Simplified DOCX processing
    else:
        text = None
    return text

if uploaded_files:
    extracted_texts = [process_file(file) for file in uploaded_files if process_file(file)]
    st.session_state.context_docs[user_id].extend(extracted_texts)
    st.sidebar.success("Documents processed and added to knowledge base!")
    st.rerun()

# ------------------ 📚 Context Management ------------------
st.sidebar.header("📚 Add Context")
context_input = st.sidebar.text_area("Enter additional context (e.g., policies, product details)")

if st.sidebar.button("Add Context"):
    if context_input:
        st.session_state.context_docs[user_id].append(context_input)
        st.sidebar.success("Context added successfully!")
        st.rerun()

# Show existing context
st.sidebar.subheader("📄 Current Context")
for i, doc in enumerate(st.session_state.context_docs[user_id]):
    st.sidebar.text(f"{i+1}. {doc[:50]}...")
    if st.sidebar.button(f"Remove {i+1}", key=f"remove_{i}"):
        del st.session_state.context_docs[user_id][i]
        st.sidebar.success("Context removed!")
        st.rerun()

# ------------------ 🔍 RAG (Retrieval-Augmented Generation) ------------------
if st.session_state.context_docs[user_id]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents(st.session_state.context_docs[user_id])
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if user_id not in st.session_state.vector_store or st.session_state.vector_store[user_id] is None:
        st.session_state.vector_store[user_id] = FAISS.from_documents(docs, embeddings)
    else:
        st.session_state.vector_store[user_id].add_documents(docs)

    retriever = st.session_state.vector_store[user_id].as_retriever()
else:
    retriever = None  # No context provided

# ------------------ 💬 Chat Interface ------------------
st.subheader("Chat")

# Display chat history
for message in st.session_state.messages[user_id]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask me anything..."):
    prompt = prompt.strip()
    if not prompt:
        st.warning("Please enter a valid question!")
        st.stop()

    st.session_state.messages[user_id].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if retriever:
        retrieval_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, retriever=retriever, memory=st.session_state.memory[user_id]
        )

        def process_input(input_text):
            return retrieval_chain.invoke({"question": input_text})

        rag_pipeline = RunnableLambda(process_input)

        with st.chat_message("assistant"):
            response_container = st.empty()
            response_text = ""

            for chunk in rag_pipeline.stream(prompt):
                response_chunk = chunk.get("answer", "")
                response_text += response_chunk
                response_container.markdown(response_text)
    else:
        response = llm.invoke(prompt)
        response_text = response.content if response else "I couldn't generate a response."

        with st.chat_message("assistant"):
            st.markdown(response_text)

    st.session_state.messages[user_id].append({"role": "assistant", "content": response_text})

# ------------------ 🗑 Clear Chat History ------------------
if st.sidebar.button("🗑 Clear Chat History"):
    st.session_state.messages[user_id] = []
    st.session_state.memory[user_id].clear()
    st.sidebar.success("Chat history cleared!")
    st.rerun()
