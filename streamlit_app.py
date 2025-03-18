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
import socket
import sqlite3
import datetime
import dropbox
import msal
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# ========================= STREAMLIT CONFIG =========================
st.set_page_config(page_title="Smart AI Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Smart AI Chatbot")

# ========================= DATABASE FOR AUTHENTICATION =========================
conn = sqlite3.connect("users.db")
cursor = conn.cursor()
cursor.execute(
    """CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT, password TEXT)"""
)
conn.commit()

# ========================= USER AUTHENTICATION =========================
st.sidebar.header("🔑 User Authentication")
auth_option = st.sidebar.radio("Login or Sign Up", ["Login", "Sign Up"])

if auth_option == "Sign Up":
    new_username = st.sidebar.text_input("Username")
    new_password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Register"):
        cursor.execute("SELECT * FROM users WHERE username=?", (new_username,))
        if cursor.fetchone():
            st.sidebar.error("Username already exists.")
        else:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (new_username, new_password))
            conn.commit()
            st.sidebar.success("✅ Registration successful! Please log in.")

elif auth_option == "Login":
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        if cursor.fetchone():
            st.sidebar.success(f"Logged in as {username}")
            st.session_state["user_id"] = username
        else:
            st.sidebar.error("🚫 Invalid credentials.")

if "user_id" not in st.session_state:
    st.stop()

# ========================= BAN USERS BY IP =========================
def get_user_ip():
    try:
        return socket.gethostbyname(socket.gethostname())
    except:
        return "Unknown"

user_ip = get_user_ip()
BANNED_IPS = ["192.168.1.100", "203.0.113.45"]

if user_ip in BANNED_IPS:
    st.error("🚫 You are banned from using this chatbot.")
    st.stop()

# ========================= CONFIGURE AI =========================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Google API key is missing! Set it as an environment variable.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)

# ========================= MULTIPLE CHAT SESSIONS =========================
# Sidebar: Manage Multiple Chats
st.sidebar.header("💬 Manage Chats")
if "chats" not in st.session_state:
    st.session_state["chats"] = {}  # Store multiple chat sessions

chat_list = list(st.session_state["chats"].keys())

# Ensure "New Chat" is always an option
chat_list.append("New Chat")

chat_name = st.sidebar.selectbox("Select a Chat", chat_list)

# If user selects "New Chat", create a new session
if chat_name == "New Chat" or chat_name not in st.session_state["chats"]:
    new_chat_name = f"Chat {len(st.session_state['chats']) + 1}"
    st.session_state["chats"][new_chat_name] = {"messages": [], "context_docs": []}
    chat_name = new_chat_name  # Set the newly created chat as active

# Initialize selected chat session
chat_session = st.session_state["chats"][chat_name]
messages = chat_session["messages"]

# ========================= FILE UPLOAD & INTEGRATION =========================
st.sidebar.header("📂 Upload & Connect Files")
uploaded_file = st.sidebar.file_uploader("Upload Local File", type=["pdf", "docx", "txt"])

if uploaded_file:
    extracted_text = docx2txt.process(uploaded_file) if uploaded_file.name.endswith(".docx") else uploaded_file.read().decode("utf-8")
    chat_session["context_docs"].append(extracted_text)
    st.sidebar.success("✅ File added!")

# ========================= SEARCH & REASON =========================
st.sidebar.header("🔍 Search & Reason")
search_query = st.sidebar.text_input("Search in Uploaded Files")

if search_query:
    results = " ".join(chat_session["context_docs"]).lower()
    st.sidebar.write(f"**Search Results:** {results[:500]}...")

# ========================= CHAT FUNCTIONALITY =========================
st.subheader(f"💬 {chat_name}")

for i, message in enumerate(messages):
    role = message["role"]
    content = message["content"]
    
    with st.chat_message(role):
        st.markdown(content)
        if st.button("📝 Edit", key=f"edit_{i}"):
            new_text = st.text_area(f"Edit message {i}", content)
            if st.button("✅ Save", key=f"save_{i}"):
                messages[i]["content"] = new_text
                st.experimental_rerun()

        if st.button("❌ Delete", key=f"delete_{i}"):
            del messages[i]
            st.experimental_rerun()

if prompt := st.chat_input("Ask me anything..."):
    messages.append({"role": "user", "content": prompt})
    response = llm.invoke(prompt)
    response_text = response.content if response else "I couldn't generate a response."

    messages.append({"role": "assistant", "content": response_text})

    with st.chat_message("assistant"):
        st.markdown(response_text)

    tts = gTTS(response_text, lang="en")
    tts.save("response.mp3")
    st.audio("response.mp3")

# ========================= DOWNLOAD CHAT HISTORY =========================
if st.sidebar.button("📄 Download Chat as PDF"):
    chat_history = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
    pdfkit.from_string(chat_history, "chat.pdf")
    with open("chat.pdf", "rb") as file:
        st.sidebar.download_button("Download PDF", file, file_name="chat_history.pdf")

# ========================= ANALYTICS DASHBOARD =========================
st.sidebar.header("📊 Chatbot Analytics")
st.sidebar.metric("Total Chats", str(len(messages)))

fig, ax = plt.subplots()
ax.bar(["Positive", "Neutral", "Negative"], [60, 30, 10])
st.sidebar.pyplot(fig)
