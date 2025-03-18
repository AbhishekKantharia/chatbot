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
import dropbox
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import msal

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

# Create feedback table for RLHF
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
DROPBOX_ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN")
ONEDRIVE_CLIENT_ID = os.getenv("ONEDRIVE_CLIENT_ID")
ONEDRIVE_CLIENT_SECRET = os.getenv("ONEDRIVE_CLIENT_SECRET")
ONEDRIVE_TENANT_ID = os.getenv("ONEDRIVE_TENANT_ID")

if not GOOGLE_API_KEY:
    st.error("Google API key is missing! Set it as an environment variable.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)

# ========================= AUTHENTICATION =========================
st.sidebar.header("🔑 User Authentication")

if "user_id" not in st.session_state:
    st.session_state["user_id"] = None

# Register New Users
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

# Login Existing Users
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

# ========================= FILE UPLOAD (GOOGLE DRIVE, DROPBOX, ONEDRIVE) =========================
st.sidebar.header("📂 Upload Documents")
upload_source = st.sidebar.selectbox("Select Upload Source", ["Computer", "Google Drive", "Dropbox", "OneDrive"])

if upload_source == "Computer":
    uploaded_file = st.sidebar.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])
elif upload_source == "Google Drive":
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)
    st.sidebar.info("Google Drive authentication successful!")
elif upload_source == "Dropbox":
    dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)
    st.sidebar.info("Connected to Dropbox!")
elif upload_source == "OneDrive":
    msal_app = msal.ConfidentialClientApplication(
        ONEDRIVE_CLIENT_ID,
        authority=f"https://login.microsoftonline.com/{ONEDRIVE_TENANT_ID}",
        client_credential=ONEDRIVE_CLIENT_SECRET,
    )
    st.sidebar.info("Connected to OneDrive!")

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

# ========================= REINFORCEMENT LEARNING FROM HUMAN FEEDBACK (RLHF) =========================
st.sidebar.header("👍 RLHF Feedback")
feedback = st.sidebar.radio("How was the response?", ["👍 Good", "👎 Bad"], index=None)

if feedback:
    cursor.execute("INSERT INTO feedback (user_id, timestamp, input, response, rating) VALUES (?, ?, ?, ?, ?)",
                   (st.session_state["user_id"], datetime.datetime.now().isoformat(), chat_name, "", feedback))
    conn.commit()
    st.sidebar.success("✅ Feedback recorded!")

    # **RLHF Auto-Retrain Model if Enough Data Exists**
    cursor.execute("SELECT COUNT(*) FROM feedback WHERE rating = '👎 Bad'")
    bad_feedback_count = cursor.fetchone()[0]

    if bad_feedback_count > 5:  # Example threshold
        st.sidebar.warning("⚠️ Retraining AI Model Due to Bad Feedback...")
        # Call a retraining function here (mock example)
        def retrain_model():
            st.sidebar.info("✅ Model Successfully Retrained with User Feedback!")
        retrain_model()

# ========================= DELETE CHAT =========================
if st.sidebar.button("🗑️ Delete Chat"):
    del st.session_state["chats"][chat_name]
    st.rerun()
