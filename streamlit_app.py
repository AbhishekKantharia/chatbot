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
import pdfkit
import socket

# ========================= CONFIGURE STREAMLIT =========================
st.set_page_config(page_title="Smart AI Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Smart AI Chatbot")

# ========================= USER MANAGEMENT =========================
def get_user_ip():
    """Fetch user IP address (for banning feature)."""
    try:
        return socket.gethostbyname(socket.gethostname())
    except:
        return "Unknown"

user_ip = get_user_ip()
BANNED_IPS = ["192.168.1.100", "203.0.113.45"]  # Add banned IPs here

if user_ip in BANNED_IPS:
    st.error("🚫 You have been banned from using this chatbot.")
    st.stop()

# ========================= MANAGE MULTIPLE CHATS =========================
st.sidebar.header("💬 Manage Chats")
if "chats" not in st.session_state:
    st.session_state["chats"] = {}  # Store multiple chat sessions

chat_list = list(st.session_state["chats"].keys()) or ["New Chat"]
chat_name = st.sidebar.selectbox("Select a Chat", chat_list)

if st.sidebar.button("➕ Start New Chat"):
    new_chat_name = f"Chat {len(st.session_state['chats']) + 1}"
    st.session_state["chats"][new_chat_name] = {"messages": []}
    chat_name = new_chat_name

if chat_name not in st.session_state["chats"]:
    st.session_state["chats"][chat_name] = {"messages": []}

chat_session = st.session_state["chats"][chat_name]
messages = chat_session["messages"]

# ========================= THEMES =========================
theme = st.sidebar.radio("🎨 Chatbot Theme", ["Light", "Dark", "Retro"])
if theme == "Dark":
    st.markdown("<style>body { background-color: black; color: white; }</style>", unsafe_allow_html=True)
elif theme == "Retro":
    st.markdown("<style>body { background-color: #f4e1d2; color: black; }</style>", unsafe_allow_html=True)

# ========================= SAVE & EXPORT CHAT =========================
def save_chat(chat_name):
    with open(f"{chat_name}.txt", "w") as file:
        for msg in messages:
            file.write(f"{msg['role'].capitalize()}: {msg['content']}\n")
    st.sidebar.success("💾 Chat saved as TXT!")

if st.sidebar.button("💾 Save Chat"):
    save_chat(chat_name)

if st.sidebar.button("📄 Export as PDF"):
    chat_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages])
    pdfkit.from_string(chat_text, f"{chat_name}.pdf")
    st.sidebar.success("✅ Chat exported as PDF!")

# ========================= DISPLAY CHAT MESSAGES =========================
st.subheader(f"💬 {chat_name}")

for i, message in enumerate(messages):
    role = message["role"]
    content = message["content"]

    with st.chat_message(role):
        col1, col2 = st.columns([0.9, 0.1])
        col1.markdown(content)

        if col2.button("❌", key=f"delete_{i}"):
            del messages[i]
            st.rerun()

# ========================= CHAT INPUT WITH COUNTER & SUGGESTIONS =========================
st.sidebar.header("💡 Suggested Prompts")
suggested_prompts = ["Tell me a joke", "Give me a fun fact", "What’s the meaning of life?"]
for prompt in suggested_prompts:
    if st.sidebar.button(prompt):
        st.session_state["input"] = prompt

input_text = st.chat_input("Type your message here... (Max 500 chars)")
if input_text:
    char_count = len(input_text)
    st.sidebar.write(f"📝 {char_count}/500 characters used")
    messages.append({"role": "user", "content": input_text})
    
    with st.chat_message("user"):
        st.markdown(input_text)

    # Simulated AI Typing Indicator
    with st.chat_message("assistant"):
        response_container = st.empty()
        response_container.markdown("🤖 AI is typing...")

    # Fake AI Response (Replace with actual AI logic if needed)
    response_text = f"🤖 AI Response: {input_text[::-1]}"  # Just reversing input for demo
    messages.append({"role": "assistant", "content": response_text})

    with st.chat_message("assistant"):
        st.markdown(response_text)

# ========================= DELETE CHAT =========================
if st.sidebar.button("🗑️ Delete Chat"):
    del st.session_state["chats"][chat_name]
    st.rerun()
