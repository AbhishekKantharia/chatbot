import os
import streamlit as st
import google.generativeai as genai
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema.runnable import RunnableLambda
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import docx2txt
import speech_recognition as sr
from gtts import gTTS
import socket

# ======================== CONFIGURE STREAMLIT ========================
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Smart AI Chatbot (Google Gemini)")

# Fetch Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Google API key is missing! Set it as an environment variable.")
    st.stop()

# Configure Google Gemini AI
genai.configure(api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)

# ======================== DOCUMENT UPLOAD (RAG) ========================
st.sidebar.header("📂 Upload Documents for RAG")
uploaded_file = st.sidebar.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])

def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif file.name.endswith(".docx"):
        return docx2txt.process(file)
    else:
        return file.read().decode("utf-8")

retriever = None
if uploaded_file:
    extracted_text = extract_text(uploaded_file)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents([extracted_text])
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever()
    st.sidebar.success("✅ Document added to chatbot knowledge!")

# ======================== DISPLAY CHAT MESSAGES ========================
st.subheader("💬 Chat")
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "👤").markdown(message["content"])

# ======================== USER INPUT ========================
prompt = st.chat_input("Ask me anything...")

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="👤").markdown(prompt)

    response_text = ""

    # Use RAG if context is available, else only AI
    if retriever:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        retrieval_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
        response_text = retrieval_chain.run(prompt)
    else:
        response_text = llm.invoke(prompt).content

    st.session_state["messages"].append({"role": "assistant", "content": response_text})
    st.chat_message("assistant", avatar="🤖").markdown(response_text)

    # Convert response to speech
    tts = gTTS(response_text, lang="en")
    tts.save("response.mp3")
    st.audio("response.mp3")

# ======================== DELETE CHAT ========================
if st.sidebar.button("🗑️ Delete Chat"):
    st.session_state["messages"] = []
    st.experimental_rerun()
