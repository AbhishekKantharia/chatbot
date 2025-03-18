import os
import streamlit as st
import sqlite3
import datetime
import google.generativeai as genai
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema.runnable import RunnableLambda
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import docx2txt
from gtts import gTTS
import matplotlib.pyplot as plt
import pdfkit

# ========================= SETUP =========================
st.set_page_config(page_title="RLHF AI Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 AI Chatbot with Reinforcement Learning from Human Feedback (RLHF)")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Google API key is missing! Set it as an environment variable.")
    st.stop()

# Configure AI Model
genai.configure(api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)

# Initialize Database for Feedback
conn = sqlite3.connect("chat_feedback.db")
c = conn.cursor()
c.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        user_input TEXT,
        bot_response TEXT,
        rating INTEGER  -- 1 = 👍, -1 = 👎
    )
""")
conn.commit()

# ========================= CHAT SESSION MANAGEMENT =========================
st.sidebar.header("💬 Manage Chats")
if "chats" not in st.session_state:
    st.session_state["chats"] = {}

chat_list = list(st.session_state["chats"].keys()) + ["New Chat"]
chat_name = st.sidebar.selectbox("Select a Chat", chat_list)

if chat_name == "New Chat" or chat_name not in st.session_state["chats"]:
    new_chat_name = f"Chat {len(st.session_state['chats']) + 1}"
    st.session_state["chats"][new_chat_name] = {"messages": [], "context_docs": []}
    chat_name = new_chat_name

chat_session = st.session_state["chats"][chat_name]
messages = chat_session["messages"]

# ========================= USER INPUT HANDLING =========================
if prompt := st.chat_input("Ask me anything..."):
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response_text = ""  # Ensure response_text is initialized

    try:
        response = llm.invoke(prompt)
        response_text = response.content if response else "I couldn't generate a response."

        with st.chat_message("assistant"):
            response_container = st.empty()
            response_container.markdown(response_text)

    except Exception as e:
        st.error(f"❌ AI Error: {str(e)}")
        response_text = "I encountered an error while generating a response."

    # Store assistant response
    messages.append({"role": "assistant", "content": response_text})

    # Generate improved voice response
    try:
        tts = gTTS(response_text, lang="en")
        tts.save("response.mp3")
        st.audio("response.mp3")
    except Exception as e:
        st.error(f"❌ Audio generation error: {str(e)}")

    # ========================= USER FEEDBACK SYSTEM (RLHF) =========================
    st.markdown("### Provide Feedback on this Response")
    col1, col2 = st.columns([0.5, 0.5])

    if col1.button("👍 Like", key=f"like_{len(messages)}"):
        c.execute("INSERT INTO feedback (user_input, bot_response, rating) VALUES (?, ?, ?)", 
                  (prompt, response_text, 1))
        conn.commit()
        st.success("Thank you for your feedback! 👍")

    if col2.button("👎 Dislike", key=f"dislike_{len(messages)}"):
        c.execute("INSERT INTO feedback (user_input, bot_response, rating) VALUES (?, ?, ?)", 
                  (prompt, response_text, -1))
        conn.commit()
        st.warning("Feedback noted. We will improve! 👎")

# ========================= ANALYTICS & CHAT HISTORY =========================
st.sidebar.header("📊 Chatbot Analytics")
c.execute("SELECT COUNT(*) FROM feedback WHERE rating = 1")
likes = c.fetchone()[0]
c.execute("SELECT COUNT(*) FROM feedback WHERE rating = -1")
dislikes = c.fetchone()[0]

st.sidebar.metric("👍 Positive Feedback", likes)
st.sidebar.metric("👎 Negative Feedback", dislikes)

fig, ax = plt.subplots()
ax.bar(["Positive", "Negative"], [likes, dislikes], color=["green", "red"])
st.sidebar.pyplot(fig)

# ========================= CLOSE DATABASE CONNECTION =========================
conn.close()
