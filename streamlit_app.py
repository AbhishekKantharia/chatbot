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
import matplotlib.pyplot as plt
import pdfkit
import sqlite3
import datetime
import socket

# Cloud storage imports
import dropbox
import msal
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# ========================= STREAMLIT CONFIG =========================
st.set_page_config(page_title="Smart AI Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Smart AI Chatbot")

# ========================= DATABASE SETUP =========================
conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()

# Create user table
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
)
""")
conn.commit()

# Create feedback table for RLHF
cursor.execute("""
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    user_input TEXT,
    bot_response TEXT,
    feedback TEXT
)
""")
conn.commit()

# ========================= USER AUTHENTICATION =========================
st.sidebar.header("🔑 User Authentication")
auth_choice = st.sidebar.radio("Login / Register", ["Login", "Register"])

if auth_choice == "Register":
    new_username = st.sidebar.text_input("Choose a username")
    new_password = st.sidebar.text_input("Choose a password", type="password")
    
    if st.sidebar.button("Register"):
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (new_username, new_password))
            conn.commit()
            st.sidebar.success("✅ Registration successful! Please log in.")
        except sqlite3.IntegrityError:
            st.sidebar.error("⚠️ Username already exists.")

elif auth_choice == "Login":
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        cursor.execute("SELECT id FROM users WHERE username = ? AND password = ?", (username, password))
        user_data = cursor.fetchone()
        if user_data:
            st.sidebar.success(f"✅ Logged in as {username}")
            st.session_state["user_id"] = user_data[0]
        else:
            st.sidebar.error("❌ Invalid credentials.")

if "user_id" not in st.session_state:
    st.stop()  # Prevents further execution until logged in

# ========================= BAN USERS BASED ON IP =========================
def get_user_ip():
    try:
        return socket.gethostbyname(socket.gethostname())
    except:
        return "Unknown"

user_ip = get_user_ip()
BANNED_IPS = ["192.168.1.100", "203.0.113.45"]

if user_ip in BANNED_IPS:
    st.error("🚫 You have been banned from using this chatbot.")
    st.stop()

# ========================= GOOGLE GEMINI AI =========================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Google API key is missing! Set it as an environment variable.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)

# ========================= MULTIPLE CHATS =========================
st.sidebar.header("💬 Manage Chats")
if "chats" not in st.session_state:
    st.session_state["chats"] = {}

chat_list = list(st.session_state["chats"].keys()) or ["New Chat"]
chat_name = st.sidebar.selectbox("Select a Chat", chat_list)

if st.sidebar.button("➕ Start New Chat"):
    new_chat_name = f"Chat {len(st.session_state['chats']) + 1}"
    st.session_state["chats"][new_chat_name] = {"messages": [], "context_docs": []}
    chat_name = new_chat_name

chat_session = st.session_state["chats"].get(chat_name, {"messages": [], "context_docs": []})
messages = chat_session["messages"]

# ========================= FILE UPLOAD & INTEGRATION =========================
st.sidebar.header("📂 Upload Documents")
uploaded_file = st.sidebar.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])

if uploaded_file:
    def extract_text(file):
        if file.name.endswith(".pdf"):
            return "".join([page.extract_text() for page in PdfReader(file).pages if page.extract_text()])
        elif file.name.endswith(".docx"):
            return docx2txt.process(file)
        else:
            return file.read().decode("utf-8")

    extracted_text = extract_text(uploaded_file)
    chat_session["context_docs"].append(extracted_text)
    st.sidebar.success("✅ Document added!")

# ========================= CHAT FUNCTIONALITY =========================
if prompt := st.chat_input("Ask me anything..."):
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response_text = llm.invoke(prompt).content
    with st.chat_message("assistant"):
        st.markdown(response_text)
    
    messages.append({"role": "assistant", "content": response_text})

# ========================= RLHF FEEDBACK =========================
st.sidebar.header("👍👎 Provide Feedback")
feedback_text = st.sidebar.text_area("Provide suggestions for improvement:")

if st.sidebar.button("Submit Feedback"):
    cursor.execute("INSERT INTO feedback (user_id, user_input, bot_response, feedback) VALUES (?, ?, ?, ?)", 
                   (st.session_state["user_id"], prompt, response_text, feedback_text))
    conn.commit()
    st.sidebar.success("✅ Feedback submitted!")

# ========================= PDF DOWNLOAD =========================
if st.sidebar.button("📄 Download Chat as PDF"):
    chat_history = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
    pdfkit.from_string(chat_history, "chat.pdf")
    with open("chat.pdf", "rb") as file:
        st.sidebar.download_button("Download PDF", file, file_name="chat_history.pdf")

# ========================= LOGOUT =========================
if st.sidebar.button("🚪 Logout"):
    st.session_state.pop("user_id", None)
    st.rerun()
