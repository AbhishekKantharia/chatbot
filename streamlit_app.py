import os
import streamlit as st
import sqlite3
import google.generativeai as genai
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema.runnable import RunnableLambda
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import docx2txt
import datetime
import socket
from gtts import gTTS
import dropbox
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import msal
import pandas as pd
import matplotlib.pyplot as plt

# Configure Streamlit
st.set_page_config(page_title="Smart AI Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Smart AI Chatbot")

# Fetch Google API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Google API key is missing! Set it as an environment variable.")
    st.stop()

# Setup AI Model
genai.configure(api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)

# Database Setup
conn = sqlite3.connect("chat_feedback.db", check_same_thread=False)
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT, password TEXT)""")
c.execute("""CREATE TABLE IF NOT EXISTS feedback (user_input TEXT, bot_response TEXT, rating INTEGER)""")
conn.commit()

# IP Banning System
BANNED_IPS = ["192.168.1.100", "203.0.113.45"]
user_ip = socket.gethostbyname(socket.gethostname())
if user_ip in BANNED_IPS:
    st.error("🚫 You have been banned from using this chatbot.")
    st.stop()

# User Authentication
st.sidebar.subheader("🔑 User Authentication")
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None

if st.session_state["user_id"]:
    st.sidebar.success(f"Logged in as {st.session_state['user_id']}")
    if st.sidebar.button("Logout"):
        st.session_state["user_id"] = None
        st.rerun()
else:
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = c.fetchone()
        if user:
            st.session_state["user_id"] = username
            st.sidebar.success(f"Welcome {username}!")
            st.rerun()
        else:
            st.sidebar.error("Invalid login credentials.")
    
    if st.sidebar.button("Sign Up"):
        c.execute("SELECT * FROM users WHERE username=?", (username,))
        if c.fetchone():
            st.sidebar.error("Username already exists.")
        else:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            st.sidebar.success("Account created! You can now log in.")

# Multi-Chat System
st.sidebar.subheader("💬 Manage Chats")
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

# File Upload & Cloud Storage
st.sidebar.subheader("📂 Upload Files")
uploaded_file = st.sidebar.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])

def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif file.name.endswith(".docx"):
        return docx2txt.process(file)
    else:
        return file.read().decode("utf-8")

if uploaded_file:
    extracted_text = extract_text(uploaded_file)
    chat_session["context_docs"].append(extracted_text)
    st.sidebar.success("✅ Document added to chatbot knowledge!")

# Chat Input
if prompt := st.chat_input("Ask me anything..."):
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response_text = ""
    if chat_session["context_docs"]:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        retrieval_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=None, memory=memory)
        rag_pipeline = RunnableLambda(lambda input_text: retrieval_chain.invoke(input_text))

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

    # Feedback System
    col1, col2 = st.columns([0.5, 0.5])
    if col1.button("👍", key=f"like_{prompt}"):
        c.execute("INSERT INTO feedback (user_input, bot_response, rating) VALUES (?, ?, 1)", (prompt, response_text))
        conn.commit()
    if col2.button("👎", key=f"dislike_{prompt}"):
        c.execute("INSERT INTO feedback (user_input, bot_response, rating) VALUES (?, ?, -1)", (prompt, response_text))
        conn.commit()

    # Audio Response
    tts = gTTS(response_text, lang="en")
    tts.save("response.mp3")
    st.audio("response.mp3")

# RLHF Analysis & AI Improvement
def analyze_feedback():
    df = pd.read_sql_query("SELECT user_input, bot_response, rating FROM feedback", conn)
    pos_feedback = df[df["rating"] == 1].shape[0]
    neg_feedback = df[df["rating"] == -1].shape[0]
    top_neg_responses = df[df["rating"] == -1]["bot_response"].value_counts().head(5)

    st.sidebar.subheader("📊 Feedback Insights")
    st.sidebar.metric("👍 Positive Feedback", pos_feedback)
    st.sidebar.metric("👎 Negative Feedback", neg_feedback)
    st.sidebar.table(top_neg_responses)

analyze_feedback()

# Delete Chat
if st.sidebar.button("🗑️ Delete Chat"):
    del st.session_state["chats"][chat_name]
    st.rerun()
