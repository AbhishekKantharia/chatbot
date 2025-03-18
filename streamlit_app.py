import streamlit as st
import sqlite3
import datetime
import os
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

# ========================= STREAMLIT CONFIG =========================
st.set_page_config(page_title="Smart AI Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Smart AI Chatbot")

# ========================= DATABASE SETUP =========================
conn = sqlite3.connect("chatbot.db", check_same_thread=False)
cursor = conn.cursor()

# Create chat history table if it does not exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    timestamp DATETIME,
    user_input TEXT,
    bot_response TEXT,
    feedback INTEGER DEFAULT NULL
)
""")
conn.commit()

# ========================= AI MODEL CONFIG =========================
llm = ChatGoogleGenerativeAI(model="gemini-pro")  # Google Gemini Model
memory = ConversationBufferMemory(memory_key="chat_history")

# ✅ Check if FAISS index exists before loading
faiss_index_path = "faiss_index"
retriever = None  # Default to None

if os.path.exists(f"{faiss_index_path}/index.faiss") and os.path.exists(f"{faiss_index_path}/index.pkl"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    retriever = FAISS.load_local(
        faiss_index_path,
        embeddings,
        allow_dangerous_deserialization=True  # ✅ Safe FAISS loading
    ).as_retriever()
else:
    st.warning("⚠️ FAISS index not found. Running without retrieval augmentation.")

# ========================= AI CHAT FUNCTION =========================
def chat_response(user_input, dynamic_prompt=None):
    """
    Generates a chatbot response using Google Gemini AI.
    Falls back to LLMChain if no retriever is available.
    """
    if dynamic_prompt:
        user_input = dynamic_prompt + "\n" + user_input

    if retriever:
        # ✅ Use ConversationalRetrievalChain if retriever exists
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory
        )
    else:
        # ✅ Use basic LLMChain if FAISS is missing
        prompt_template = PromptTemplate(
            input_variables=["question"],
            template="You are an AI assistant. Answer this question: {question}"
        )
        chain = LLMChain(llm=llm, prompt=prompt_template)

    return chain.run(user_input)

# ========================= DISPLAY CHAT HISTORY =========================
st.sidebar.header("📝 Chat History")
cursor.execute("SELECT user_input, bot_response FROM chat_history ORDER BY timestamp DESC LIMIT 5")
chat_history = cursor.fetchall()

for user_msg, bot_msg in chat_history[::-1]:  # Reverse to display newest last
    with st.sidebar.expander(f"🗨️ {user_msg}", expanded=False):
        st.write(f"🤖 {bot_msg}")

# ========================= USER INPUT =========================
user_input = st.text_input("You:", "")

if user_input:
    # ✅ Adaptive Prompting based on feedback trends 📊
    cursor.execute("SELECT COUNT(*) FROM chat_history WHERE feedback = 1")
    positive_feedback_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM chat_history WHERE feedback = 0")
    negative_feedback_count = cursor.fetchone()[0]

    adaptive_prompt = "Use a more engaging tone." if positive_feedback_count > negative_feedback_count else "Provide more detailed explanations."

    bot_response = chat_response(user_input, adaptive_prompt)

    # ✅ Display chat
    st.write(f"**You:** {user_input}")
    st.write(f"**AI:** {bot_response}")

    # ✅ Store conversation
    cursor.execute(
        "INSERT INTO chat_history (user_id, timestamp, user_input, bot_response) VALUES (?, ?, ?, ?)",
        (st.session_state.get("user_id", "anonymous"), datetime.datetime.now(), user_input, bot_response)
    )
    conn.commit()

    # ========================= FEEDBACK BUTTONS =========================
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍", key=f"positive_{user_input}"):
            cursor.execute("UPDATE chat_history SET feedback = 1 WHERE user_input = ?", (user_input,))
            conn.commit()
            st.success("Thanks for your feedback! ✅")

    with col2:
        if st.button("👎", key=f"negative_{user_input}"):
            cursor.execute("UPDATE chat_history SET feedback = 0 WHERE user_input = ?", (user_input,))
            conn.commit()
            st.error("We'll improve this response! 🔄")

# ========================= DELETE CHAT =========================
if st.sidebar.button("🗑️ Delete Chat"):
    cursor.execute("DELETE FROM chat_history")
    conn.commit()
    st.session_state["chat_history"] = []
    st.rerun()
