import os
import sqlite3
import streamlit as st
import google.generativeai as genai
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
if "username" not in st.session_state:
    st.session_state["username"] = ""

if not st.session_state["user_id"]:
    st.sidebar.header("🔐 Login / Register")
    auth_option = st.sidebar.radio("Select Option", ["Login", "Register"])
    username = st.sidebar.text_input("Username", key="username_input")
    password = st.sidebar.text_input("Password", type="password", key="password_input")

    if auth_option == "Login":
        if st.sidebar.button("Login"):
            cursor.execute("SELECT id, username FROM users WHERE username = ? AND password = ?", (username, password))
            user = cursor.fetchone()
            if user:
                st.session_state["user_id"] = user[0]
                st.session_state["username"] = user[1]
                st.rerun()  # ✅ FIXED: `st.experimental_rerun()` replaced with `st.rerun()`
            else:
                st.sidebar.error("Invalid credentials.")

    elif auth_option == "Register":
        if st.sidebar.button("Register"):
            try:
                cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
                conn.commit()
                st.sidebar.success("✅ Account created! Please log in.")
                st.rerun()
            except sqlite3.IntegrityError:
                st.sidebar.error("🚫 Username already exists.")

else:
    st.sidebar.success(f"Logged in as {st.session_state['username']}")
    if st.sidebar.button("Logout"):
        st.session_state["user_id"] = None
        st.session_state["username"] = ""
        st.rerun()  # ✅ FIXED: Using `st.rerun()` instead of `st.experimental_rerun()`

# 🎯 **Google AI Configuration**
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Google API Key is missing! Set it as an environment variable.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)
llm = genai.GenerativeModel("gemini-1.5-pro")

# 🎯 **Save Chat History to SQLite**
def save_chat(user_id, user_input, bot_response):
    cursor.execute("INSERT INTO chats (user_id, timestamp, user_input, bot_response) VALUES (?, ?, ?, ?)", 
                   (user_id, datetime.datetime.now(), user_input, bot_response))
    conn.commit()

# 🎯 **Retrieve Chat History**
def get_chat_history(user_id):
    cursor.execute("SELECT user_input, bot_response FROM chats WHERE user_id = ? ORDER BY timestamp", (user_id,))
    chats = cursor.fetchall()
    return [{"role": "user", "content": row[0]} for row in chats] + \
           [{"role": "assistant", "content": row[1]} for row in chats]

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

        response = llm.generate_content(prompt)
        response_text = response.text if response else "I couldn't generate a response."

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
    total_chats = len(messages) // 2  # Since each interaction has user & bot response

    st.sidebar.metric("Total Chats", total_chats)

    fig, ax = plt.subplots()
    ax.bar(["Total Chats"], [total_chats])
    st.sidebar.pyplot(fig)
