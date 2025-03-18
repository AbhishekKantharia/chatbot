import os
import streamlit as st
import random
import time
import pdfkit
import socket
from textstat import flesch_reading_ease

# ========================= CONFIGURE STREAMLIT =========================
st.set_page_config(page_title="Smart AI Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Smart AI Chatbot")

# ========================= GET USER IP (BAN FEATURE) =========================
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

# ========================= MANAGE MULTIPLE CHATS =========================
st.sidebar.header("💬 Manage Chats")
if "chats" not in st.session_state:
    st.session_state["chats"] = {}

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

# ========================= THEMES & CUSTOMIZATION =========================
theme = st.sidebar.radio("🎨 Chatbot Theme", ["Light", "Dark", "Retro", "Cyberpunk"])
if theme == "Dark":
    st.markdown("<style>body { background-color: black; color: white; }</style>", unsafe_allow_html=True)
elif theme == "Retro":
    st.markdown("<style>body { background-color: #f4e1d2; color: black; }</style>", unsafe_allow_html=True)
elif theme == "Cyberpunk":
    st.markdown("<style>body { background-color: #080c1b; color: #0ff; }</style>", unsafe_allow_html=True)

font = st.sidebar.selectbox("🖋️ Choose Font", ["Default", "Serif", "Monospace"])
if font == "Serif":
    st.markdown("<style>body { font-family: serif; }</style>", unsafe_allow_html=True)
elif font == "Monospace":
    st.markdown("<style>body { font-family: monospace; }</style>", unsafe_allow_html=True)

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
        col1, col2 = st.columns([0.85, 0.15])
        col1.markdown(content)

        if col2.button("📝", key=f"edit_{i}"):
            new_content = st.text_area(f"Edit message {i}", content)
            if st.button("✅ Save", key=f"save_{i}"):
                messages[i]["content"] = new_content
                st.rerun()

        if col2.button("❌", key=f"delete_{i}"):
            del messages[i]
            st.rerun()

# ========================= CHAT INPUT =========================
input_text = st.chat_input("Type your message here... (Max 500 chars)")
if input_text:
    char_count = len(input_text)
    st.sidebar.write(f"📝 {char_count}/500 characters used")
    messages.append({"role": "user", "content": input_text})

    with st.chat_message("user"):
        st.markdown(input_text)

    # Typing Indicator Simulation
    with st.chat_message("assistant"):
        response_container = st.empty()
        response_container.markdown("🤖 AI is typing...")
        time.sleep(1.5)

    # AI Mood-Based Responses
    if any(word in input_text.lower() for word in ["happy", "excited", "great"]):
        response_text = "😊 That’s amazing to hear!"
    elif any(word in input_text.lower() for word in ["sad", "depressed", "bad"]):
        response_text = "😢 I’m here for you. Want to talk about it?"
    else:
        response_text = f"🤖 AI Response: {input_text[::-1]}"  # Simple reverse response for demo

    messages.append({"role": "assistant", "content": response_text})

    with st.chat_message("assistant"):
        st.markdown(response_text)

# ========================= TEXT ANALYSIS =========================
if st.sidebar.button("📊 Analyze Text"):
    total_words = sum(len(msg["content"].split()) for msg in messages)
    total_sentences = sum(msg["content"].count(".") for msg in messages)
    readability_score = flesch_reading_ease("\n".join([msg["content"] for msg in messages]))
    
    st.sidebar.write(f"📝 Total Words: {total_words}")
    st.sidebar.write(f"📖 Total Sentences: {total_sentences}")
    st.sidebar.write(f"📚 Readability Score: {round(readability_score, 2)}")

# ========================= FUN EXTRAS =========================
# 🎭 Easter Eggs
easter_eggs = {
    "hello there": "General Kenobi! ⚔️",
    "i love you": "Awww, I love you too! 💙",
    "open the pod bay doors": "I’m sorry, Dave. I’m afraid I can’t do that. 🚀"
}

if input_text.lower() in easter_eggs:
    with st.chat_message("assistant"):
        st.markdown(easter_eggs[input_text.lower()])

# 🧠 Random Quote Generator
if st.sidebar.button("💬 Random Quote"):
    quotes = ["Believe in yourself!", "Stay positive!", "You got this!", "Every day is a second chance."]
    st.sidebar.success(random.choice(quotes))

# 🧐 Daily Fun Fact
if st.sidebar.button("📅 Fun Fact"):
    facts = ["Did you know honey never spoils?", "Bananas are berries, but strawberries aren't!", "Octopuses have three hearts."]
    st.sidebar.success(random.choice(facts))

# ========================= DELETE CHAT =========================
if st.sidebar.button("🗑️ Delete Chat"):
    del st.session_state["chats"][chat_name]
    st.rerun()
