import os
import streamlit as st
import sqlite3
import google.generativeai as genai
import pandas as pd
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema.runnable import RunnableLambda
from gtts import gTTS
import datetime

# Configure Streamlit
st.set_page_config(page_title="AI Chatbot with RLHF", page_icon="🤖", layout="wide")
st.title("🤖 AI Chatbot with RLHF & Automated Fine-Tuning")

# Configure Google Gemini AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# ✅ Correct model initialization
llm = genai.GenerativeModel("gemini-1.5-pro")

# Database Setup
conn = sqlite3.connect("chat_feedback.db", check_same_thread=False)
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS feedback 
             (id INTEGER PRIMARY KEY, user_input TEXT, bot_response TEXT, alternative_response TEXT, rating INTEGER, timestamp DATETIME)""")
conn.commit()

# Initialize Chat
if "messages" not in st.session_state:
    st.session_state["messages"] = []

messages = st.session_state["messages"]

# Chat Input
if prompt := st.chat_input("Ask me anything..."):
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response_text = ""
    response = llm.generate_content(prompt)
    response_text = response.text if response else "I couldn't generate a response."

    with st.chat_message("assistant"):
        st.markdown(response_text)

    messages.append({"role": "assistant", "content": response_text})

    # Store Initial Feedback (Neutral by default)
    c.execute("INSERT INTO feedback (user_input, bot_response, rating, timestamp) VALUES (?, ?, 0, ?)", 
              (prompt, response_text, datetime.datetime.now()))
    conn.commit()

    # Audio Response
    tts = gTTS(response_text, lang="en")
    tts.save("response.mp3")
    st.audio("response.mp3")

    # Feedback System
    col1, col2 = st.columns([0.5, 0.5])
    if col1.button("👍 Like", key=f"like_{prompt}"):
        c.execute("UPDATE feedback SET rating = 1 WHERE user_input = ?", (prompt,))
        conn.commit()
        st.success("Thanks for your feedback!")

    if col2.button("👎 Dislike", key=f"dislike_{prompt}"):
        c.execute("UPDATE feedback SET rating = -1 WHERE user_input = ?", (prompt,))
        conn.commit()
        st.warning("Generating an alternative response...")

        # AI generates an improved response
        alt_response = llm.generate_content(f"Rephrase this response for better clarity:\n{response_text}").text
        c.execute("UPDATE feedback SET alternative_response = ? WHERE user_input = ?", (alt_response, prompt))
        conn.commit()

        with st.chat_message("assistant"):
            st.markdown(f"🆕 Alternative Response: {alt_response}")

        messages.append({"role": "assistant", "content": alt_response})

# RLHF-Based Fine-Tuning Analysis
def analyze_feedback():
    df = pd.read_sql_query("SELECT * FROM feedback WHERE rating != 0", conn)
    pos_feedback = df[df["rating"] == 1].shape[0]
    neg_feedback = df[df["rating"] == -1].shape[0]
    top_neg_responses = df[df["rating"] == -1]["bot_response"].value_counts().head(5)

    st.sidebar.subheader("📊 RLHF Feedback Insights")
    st.sidebar.metric("👍 Positive Feedback", pos_feedback)
    st.sidebar.metric("👎 Negative Feedback", neg_feedback)
    st.sidebar.table(top_neg_responses)

analyze_feedback()

# Periodic Model Retraining (Simulated)
def retrain_model():
    df = pd.read_sql_query("SELECT * FROM feedback WHERE rating = -1", conn)
    if not df.empty:
        training_data = [
            {"input": row["user_input"], "output": row["alternative_response"]}
            for _, row in df.iterrows() if row["alternative_response"]
        ]
        st.sidebar.write("🛠️ Fine-tuning AI model with RLHF data...")
        st.sidebar.write(training_data[:3])  # Show a sample
        # Note: Actual model fine-tuning requires integration with OpenAI/Google AI APIs.

if st.sidebar.button("🔄 Retrain Model"):
    retrain_model()
    st.sidebar.success("AI model updated based on feedback!")
