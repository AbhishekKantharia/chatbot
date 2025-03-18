import os
import streamlit as st
import google.generativeai as genai
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import docx2txt
import pdfkit
import socket
import sqlite3
import datetime
import dropbox
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from onedrivesdk import OneDriveClient

# Configure Streamlit
st.set_page_config(page_title="Smart AI Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Smart AI Chatbot")

# Database Setup for Users & Feedback
conn = sqlite3.connect("chatbot.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    password TEXT)''')

cursor.execute('''CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    timestamp DATETIME,
                    user_input TEXT,
                    bot_response TEXT,
                    rating INTEGER)''')

conn.commit()

# User Authentication
st.sidebar.header("🔐 User Authentication")
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None

auth_mode = st.sidebar.radio("Login or Register", ["Login", "Register"])

if auth_mode == "Register":
    new_username = st.sidebar.text_input("Choose a Username")
    new_password = st.sidebar.text_input("Choose a Password", type="password")
    
    if st.sidebar.button("Register"):
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (new_username, new_password))
            conn.commit()
            st.sidebar.success("Account created! Please log in.")
        except sqlite3.IntegrityError:
            st.sidebar.error("Username already exists.")

if auth_mode == "Login":
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        cursor.execute("SELECT id FROM users WHERE username = ? AND password = ?", (username, password))
        user = cursor.fetchone()
        if user:
            st.session_state["user_id"] = user[0]
            st.sidebar.success(f"Logged in as {username}")
            st.experimental_rerun()
        else:
            st.sidebar.error("Invalid credentials.")

if st.session_state["user_id"]:
    if st.sidebar.button("Logout"):
        st.session_state["user_id"] = None
        st.experimental_rerun()

# Configure Google Gemini AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Google API key is missing! Set it as an environment variable.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)

# Sidebar: Manage Multiple Chats
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

# Sidebar: File Upload
st.sidebar.header("📂 Upload Files")
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

# RAG Setup
retriever = None
if chat_session["context_docs"]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents(chat_session["context_docs"])
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever()

# Display chat messages
st.subheader(f"💬 {chat_name}")
for i, message in enumerate(messages):
    role = message["role"]
    content = message["content"]

    with st.chat_message(role):
        st.markdown(content)

# Chat Input
if prompt := st.chat_input("Ask me anything..."):
    messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    response_text = llm.invoke(prompt).content
    messages.append({"role": "assistant", "content": response_text})

    with st.chat_message("assistant"):
        st.markdown(response_text)

    # Feedback
    rating = st.slider("Rate this response", 1, 5, key=f"rate_{len(messages)}")
    if st.button("Submit Feedback", key=f"feedback_{len(messages)}"):
        cursor.execute("INSERT INTO feedback (user_id, timestamp, user_input, bot_response, rating) VALUES (?, ?, ?, ?, ?)",
                       (st.session_state["user_id"], datetime.datetime.now(), prompt, response_text, rating))
        conn.commit()
        st.success("Feedback submitted!")

# Delete chat
if st.sidebar.button("🗑️ Delete Chat"):
    del st.session_state["chats"][chat_name]
    st.rerun()
