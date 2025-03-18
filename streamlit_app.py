import os
import sqlite3
import streamlit as st
import google.generativeai as genai
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import docx2txt
from gtts import gTTS
import datetime
import matplotlib.pyplot as plt

# 🎯 **SQLite Database Setup**
conn = sqlite3.connect("chatbot.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS chats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    timestamp TEXT,
    user_input TEXT,
    bot_response TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id)
)
""")
conn.commit()

# 🎯 **Streamlit UI Configuration**
st.set_page_config(page_title="Smart AI Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Smart AI Chatbot")

# 🎯 **User Authentication**
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None

if not st.session_state["user_id"]:
    st.sidebar.header("🔐 Login / Register")
    auth_option = st.sidebar.radio("Select Option", ["Login", "Register"])
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if auth_option == "Login":
        if st.sidebar.button("Login"):
            cursor.execute("SELECT id FROM users WHERE username = ? AND password = ?", (username, password))
            user = cursor.fetchone()
            if user:
                st.session_state["user_id"] = user[0]
                st.experimental_rerun()
            else:
                st.sidebar.error("Invalid credentials.")

    elif auth_option == "Register":
        if st.sidebar.button("Register"):
            try:
                cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
                conn.commit()
                st.sidebar.success("Account created! Please log in.")
                st.experimental_rerun()
            except sqlite3.IntegrityError:
                st.sidebar.error("Username already exists.")

else:
    st.sidebar.success(f"Logged in as {username}")
    if st.sidebar.button("Logout"):
        st.session_state["user_id"] = None
        st.experimental_rerun()

# 🎯 **Google AI Configuration**
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)

# 🎯 **Save Chat History to SQLite**
def save_chat(user_id, user_input, bot_response):
    cursor.execute("INSERT INTO chats (user_id, timestamp, user_input, bot_response) VALUES (?, ?, ?, ?)", 
                   (user_id, datetime.datetime.now(), user_input, bot_response))
    conn.commit()

# 🎯 **Retrieve Chat History**
def get_chat_history(user_id):
    cursor.execute("SELECT user_input, bot_response FROM chats WHERE user_id = ? ORDER BY timestamp", (user_id,))
    return [{"role": "user", "content": row[0]} for row in cursor.fetchall()] + \
           [{"role": "assistant", "content": row[1]} for row in cursor.fetchall()]

# 🎯 **Sentiment Analysis**
def analyze_sentiment(text):
    if "good" in text or "happy" in text:
        return "Positive"
    elif "bad" in text or "sad" in text:
        return "Negative"
    else:
        return "Neutral"

# 🎯 **Chat Interface**
if st.session_state["user_id"]:
    messages = get_chat_history(st.session_state["user_id"])
    st.subheader("💬 Chat")

    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me anything..."):
        messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response = llm.invoke(prompt)
        response_text = response.content if response else "I couldn't generate a response."

        with st.chat_message("assistant"):
            st.markdown(response_text)

        # Store chat in SQLite
        save_chat(st.session_state["user_id"], prompt, response_text)

        # 🎯 **Audio Response**
        try:
            tts = gTTS(response_text, lang="en")
            tts.save("response.mp3")
            st.audio("response.mp3")
        except Exception as e:
            st.error(f"❌ Audio error: {str(e)}")

# 🎯 **Analytics Dashboard**
if st.session_state["user_id"]:
    st.sidebar.header("📊 Analytics")
    messages = get_chat_history(st.session_state["user_id"])
    total_chats = len(messages)
    positive_count = sum(1 for m in messages if analyze_sentiment(m["content"]) == "Positive")
    negative_count = sum(1 for m in messages if analyze_sentiment(m["content"]) == "Negative")
    neutral_count = total_chats - (positive_count + negative_count)

    st.sidebar.metric("Total Chats", total_chats)
    st.sidebar.metric("Positive Responses", positive_count)
    st.sidebar.metric("Negative Responses", negative_count)

    fig, ax = plt.subplots()
    ax.bar(["Positive", "Neutral", "Negative"], [positive_count, neutral_count, negative_count])
    st.sidebar.pyplot(fig)
