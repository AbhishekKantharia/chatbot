import streamlit as st
import sqlite3
import datetime
import random
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import docx2txt

# ========================= STREAMLIT CONFIG =========================
st.set_page_config(page_title="Smart AI Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Smart AI Chatbot with RLHF")

# ========================= DATABASE CONFIGURATION =========================
conn = sqlite3.connect("chatbot.db", check_same_thread=False)
cursor = conn.cursor()

# User Authentication Table
cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )
""")

# Chat History Table
cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        timestamp DATETIME,
        user_input TEXT,
        bot_response TEXT,
        feedback INTEGER DEFAULT NULL
    )
""")
conn.commit()

# ========================= USER AUTHENTICATION =========================
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None

def login(username, password):
    cursor.execute("SELECT id FROM users WHERE username = ? AND password = ?", (username, password))
    user = cursor.fetchone()
    return user[0] if user else None

def register(username, password):
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

# ========================= USER LOGIN & REGISTRATION =========================
st.sidebar.subheader("User Authentication")
auth_choice = st.sidebar.radio("Select an option:", ["Login", "Register"])

if auth_choice == "Login":
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        user_id = login(username, password)
        if user_id:
            st.session_state["user_id"] = user_id
            st.sidebar.success(f"Logged in as {username}")
            st.rerun()
        else:
            st.sidebar.error("Invalid credentials")

elif auth_choice == "Register":
    new_username = st.sidebar.text_input("New Username")
    new_password = st.sidebar.text_input("New Password", type="password")
    if st.sidebar.button("Register"):
        if register(new_username, new_password):
            st.sidebar.success("Account created successfully! Please login.")
        else:
            st.sidebar.error("Username already exists.")

# ========================= FILE UPLOAD =========================
st.sidebar.subheader("Upload a File")
uploaded_file = st.sidebar.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])

if uploaded_file:
    file_extension = uploaded_file.name.split(".")[-1]

    if file_extension == "pdf":
        pdf_reader = PdfReader(uploaded_file)
        text = "\n".join([page.extract_text() for page in pdf_reader.pages])
    elif file_extension == "docx":
        text = docx2txt.process(uploaded_file)
    else:
        text = uploaded_file.read().decode("utf-8")

    st.session_state["uploaded_text"] = text
    st.sidebar.success("File uploaded successfully!")

# ========================= CHATBOT SESSION =========================
if "chats" not in st.session_state:
    st.session_state["chats"] = {"New Chat": {"messages": [], "context": None}}

chat_name = st.sidebar.selectbox("Select a chat:", list(st.session_state["chats"].keys()))
chat_session = st.session_state["chats"][chat_name]
messages = chat_session["messages"]

st.subheader("Chat with AI")

for idx, message in enumerate(messages):
    with st.chat_message(message["role"]):
        st.write(message["content"])

    # Add feedback buttons
    if message["role"] == "assistant":
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"👍 Like {idx}", key=f"like_{idx}"):
                cursor.execute("UPDATE chat_history SET feedback = 1 WHERE id = ?", (message["id"],))
                conn.commit()
                st.success("Feedback saved!")
        with col2:
            if st.button(f"👎 Dislike {idx}", key=f"dislike_{idx}"):
                cursor.execute("UPDATE chat_history SET feedback = -1 WHERE id = ?", (message["id"],))
                conn.commit()
                st.warning("Feedback saved!")

user_input = st.chat_input("Type your message...")

if user_input:
    # Placeholder AI Response - Replace this with an actual AI model
    ai_responses = [
        f"AI Response: {user_input[::-1]}",
        f"Here's a fun fact: {user_input.capitalize()}",
        f"{user_input}... but what if it was said backward? {user_input[::-1]}",
    ]
    
    # Simple RLHF Reward-Based Learning: If more positive feedback is given, prioritize the best responses.
    cursor.execute("SELECT COUNT(*) FROM chat_history WHERE feedback = 1")
    positive_feedback_count = cursor.fetchone()[0]
    
    # If we have positive feedback, prefer the first response, otherwise randomize
    response_text = ai_responses[0] if positive_feedback_count > 5 else random.choice(ai_responses)

    messages.append({"role": "user", "content": user_input})
    messages.append({"role": "assistant", "content": response_text})

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        st.write(response_text)

    # Save chat history
    cursor.execute(
        "INSERT INTO chat_history (user_id, timestamp, user_input, bot_response) VALUES (?, ?, ?, ?)",
        (st.session_state["user_id"], datetime.datetime.now(), user_input, response_text),
    )
    conn.commit()

# ========================= DELETE CHAT =========================
if st.sidebar.button("🗑️ Delete Chat"):
    del st.session_state["chats"][chat_name]
    st.rerun()
