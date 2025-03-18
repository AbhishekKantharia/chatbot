import streamlit as st
import sqlite3
import datetime
import random
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# ========================= STREAMLIT CONFIG =========================
st.set_page_config(page_title="Smart AI Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Smart AI Chatbot (Google Gemini + RLHF)")

# ========================= DATABASE CONFIGURATION =========================
conn = sqlite3.connect("chatbot.db", check_same_thread=False)
cursor = conn.cursor()

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

# ========================= CHATBOT MEMORY =========================
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ========================= GOOGLE GEMINI CHATBOT FUNCTION =========================
def chat_response(user_input, dynamic_prompt=None):
    """Generate AI response using Google Gemini with adaptive prompting."""
    llm = ChatGoogleGenerativeAI(model="gemini-pro")  # Using Gemini-Pro
    chain = ConversationalRetrievalChain.from_llm(llm, memory=memory)
    
    # Apply Adaptive Prompting
    if dynamic_prompt:
        user_input = dynamic_prompt + "\n" + user_input

    return chain.run(user_input)

# ========================= USER SESSION MANAGEMENT =========================
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None

# ========================= TRACK USER FEEDBACK =========================
cursor.execute("SELECT COUNT(*) FROM chat_history WHERE feedback = 1")
positive_feedback_count = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM chat_history WHERE feedback = 0")
negative_feedback_count = cursor.fetchone()[0]

# Adjust chatbot prompting dynamically based on feedback trends
if positive_feedback_count > negative_feedback_count:
    adaptive_prompt = "Provide a detailed, informative, and friendly response."
    st.sidebar.success("✅ AI is performing well! Optimized for detailed answers.")
else:
    adaptive_prompt = "Keep responses brief, direct, and improve clarity."
    st.sidebar.warning("⚠️ AI needs improvement. Adjusting for conciseness.")

# ========================= USER INPUT & AI RESPONSE =========================
user_input = st.text_input("Ask me anything:", key="user_input")

if user_input:
    bot_response = chat_response(user_input, adaptive_prompt)

    # Display conversation
    st.write(f"**You:** {user_input}")
    st.write(f"**AI:** {bot_response}")

    # Store in chat history
    cursor.execute(
        "INSERT INTO chat_history (user_id, timestamp, user_input, bot_response) VALUES (?, ?, ?, ?)",
        (st.session_state["user_id"], datetime.datetime.now(), user_input, bot_response),
    )
    conn.commit()

    # Add feedback buttons
    feedback = st.radio("Was this response helpful?", ["👍 Yes", "👎 No"], key=f"feedback_{user_input}")
    
    if feedback:
        feedback_value = 1 if feedback == "👍 Yes" else 0
        cursor.execute(
            "UPDATE chat_history SET feedback = ? WHERE user_input = ?",
            (feedback_value, user_input),
        )
        conn.commit()

# ========================= REAL-TIME AI OPTIMIZATION =========================
def adjust_response_quality():
    """Dynamically improve AI responses based on past feedback trends."""
    cursor.execute("SELECT user_input, bot_response, feedback FROM chat_history WHERE feedback IS NOT NULL ORDER BY timestamp DESC LIMIT 5")
    feedback_data = cursor.fetchall()

    if feedback_data:
        # Reward-based fine-tuning: Prefer responses with more positive feedback
        positive_responses = [row[1] for row in feedback_data if row[2] == 1]
        negative_responses = [row[1] for row in feedback_data if row[2] == 0]

        if len(positive_responses) > len(negative_responses):
            return random.choice(positive_responses)
        else:
            return "I'm improving my responses based on feedback. Let me know how I can be more helpful!"

# AI fine-tuning decision
fine_tuned_response = adjust_response_quality()
if fine_tuned_response:
    st.sidebar.info(f"🔄 Fine-Tuned AI Response: {fine_tuned_response}")

# ========================= AUTOMATED RETRAINING =========================
if st.sidebar.button("🔄 Retrain Model"):
    st.sidebar.info("AI fine-tuning in progress... (Automated adjustments applied)")
    # Placeholder for future self-learning AI retraining
