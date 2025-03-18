import os
import streamlit as st
import google.generativeai as genai
import sqlite3
import datetime
import socket
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema.runnable import RunnableLambda
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import docx2txt
from gtts import gTTS
import pdfkit

# ========================= STREAMLIT CONFIG =========================
st.set_page_config(page_title="Smart AI Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Smart AI Chatbot")

# ========================= DATABASE SETUP =========================
conn = sqlite3.connect("chatbot.db", check_same_thread=False)
cursor = conn.cursor()

# Create user table
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
""")

# Create feedback table
cursor.execute("""
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    timestamp DATETIME,
    input TEXT,
    response TEXT,
    rating TEXT
)
""")

conn.commit()

# ========================= GOOGLE API CONFIG =========================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Google API key is missing! Set it as an environment variable.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)

# ========================= USER AUTHENTICATION =========================
st.sidebar.header("🔑 User Authentication")

if "user_id" not in st.session_state:
    st.session_state["user_id"] = None

# Register new users
if st.sidebar.button("Register New User"):
    st.session_state["auth_mode"] = "register"

if "auth_mode" not in st.session_state:
    st.session_state["auth_mode"] = "login"

if st.session_state["auth_mode"] == "register":
    new_username = st.sidebar.text_input("Choose a Username")
    new_password = st.sidebar.text_input("Choose a Password", type="password")
    
    if st.sidebar.button("Create Account"):
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (new_username, new_password))
            conn.commit()
            st.sidebar.success("✅ Account created! Please log in.")
            st.session_state["auth_mode"] = "login"
        except sqlite3.IntegrityError:
            st.sidebar.error("⚠️ Username already exists!")

# Login existing users
if st.session_state["auth_mode"] == "login":
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = cursor.fetchone()

        if user:
            st.session_state["user_id"] = user[1]  # Store username as user_id
            st.sidebar.success(f"Logged in as {username}")
        else:
            st.sidebar.error("⚠️ Invalid credentials!")

if st.session_state["user_id"] is None:
    st.stop()
else:
    user_id = st.session_state["user_id"]
    st.sidebar.success(f"Logged in as {user_id}")
    
    if st.sidebar.button("Logout"):
        st.session_state["user_id"] = None
        st.rerun()

# ========================= CHAT MANAGEMENT =========================
st.sidebar.header("💬 Manage Chats")
if "chats" not in st.session_state:
    st.session_state["chats"] = {}

chat_list = list(st.session_state["chats"].keys()) or ["New Chat"]
chat_name = st.sidebar.selectbox("Select a Chat", chat_list)

if st.sidebar.button("➕ Start New Chat"):
    new_chat_name = f"Chat {len(st.session_state['chats']) + 1}"
    st.session_state["chats"][new_chat_name] = {"messages": [], "context_docs": []}
    chat_name = new_chat_name

if chat_name not in st.session_state["chats"]:
    st.session_state["chats"][chat_name] = {"messages": [], "context_docs": []}

chat_session = st.session_state["chats"][chat_name]
messages = chat_session["messages"]

# ========================= FILE UPLOAD FOR RAG =========================
st.sidebar.header("📂 Upload Documents for RAG")
uploaded_file = st.sidebar.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])

if uploaded_file:
    def extract_text(file):
        if file.name.endswith(".pdf"):
            reader = PdfReader(file)
            return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif file.name.endswith(".docx"):
            return docx2txt.process(file)
        else:
            return file.read().decode("utf-8")

    extracted_text = extract_text(uploaded_file)
    if extracted_text:
        chat_session["context_docs"].append(extracted_text)
        st.sidebar.success("✅ Document added to chatbot knowledge!")

retriever = None
if chat_session["context_docs"]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents(chat_session["context_docs"])
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever()

# ========================= CHAT UI =========================
st.subheader(f"💬 {chat_name}")
for message in messages:
    st.chat_message(message["role"]).markdown(message["content"])

if prompt := st.chat_input("Ask me anything..."):
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response_text = ""

    if retriever:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        retrieval_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
        rag_pipeline = RunnableLambda(lambda x: retrieval_chain.invoke(x))

        with st.chat_message("assistant"):
            response_container = st.empty()
            for chunk in rag_pipeline.stream(prompt):
                response_text += chunk.text if hasattr(chunk, "text") else chunk
            response_container.markdown(response_text)

    else:
        response = llm.invoke(prompt)
        response_text = response.content if response else "I couldn't generate a response."

        with st.chat_message("assistant"):
            st.markdown(response_text)

    messages.append({"role": "assistant", "content": response_text})

    # Generate audio response
    tts = gTTS(response_text, lang="en")
    tts.save("response.mp3")
    st.audio("response.mp3")

# ========================= RLHF FEEDBACK SYSTEM =========================
st.sidebar.header("👍 RLHF Feedback")
feedback = st.sidebar.radio("How was the response?", ["👍 Good", "👎 Bad"], index=None)

if feedback:
    cursor.execute("INSERT INTO feedback (user_id, timestamp, input, response, rating) VALUES (?, ?, ?, ?, ?)",
                   (user_id, datetime.datetime.now().isoformat(), prompt, response_text, feedback))
    conn.commit()
    st.sidebar.success("✅ Feedback recorded!")

# ========================= DELETE CHAT =========================
if st.sidebar.button("🗑️ Delete Chat"):
    del st.session_state["chats"][chat_name]
    st.rerun()
