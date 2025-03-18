import os
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from PyPDF2 import PdfReader
import docx2txt
import sqlite3
import socket

# =========================== CONFIGURATION =========================== #
st.set_page_config(page_title="Smart AI Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Smart AI Chatbot")

# =========================== DATABASE SETUP =========================== #
conn = sqlite3.connect("chat_history.db")
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_input TEXT,
        ai_response TEXT,
        feedback INTEGER
    )
""")
conn.commit()

# =========================== IP BANNING SYSTEM =========================== #
BANNED_IPS = ["192.168.1.100", "203.0.113.45"]  # Add banned IPs here
def get_user_ip():
    try:
        return socket.gethostbyname(socket.gethostname())
    except:
        return "Unknown"
if get_user_ip() in BANNED_IPS:
    st.error("🚫 You have been banned from using this chatbot.")
    st.stop()

# =========================== AI MODEL SETUP =========================== #
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Google API key is missing! Set it as an environment variable.")
    st.stop()
    
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)

# =========================== CHAT SESSION MANAGEMENT =========================== #
st.sidebar.header("💬 Manage Chats")
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

# =========================== DOCUMENT UPLOAD FOR RAG =========================== #
st.sidebar.header("📂 Upload Documents for RAG")
uploaded_file = st.sidebar.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])
if uploaded_file:
    def extract_text(file):
        try:
            if file.name.endswith(".pdf"):
                reader = PdfReader(file)
                return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
            elif file.name.endswith(".docx"):
                return docx2txt.process(file)
            else:
                return file.read().decode("utf-8")
        except Exception as e:
            st.sidebar.error(f"Error processing file: {str(e)}")
            return ""
    extracted_text = extract_text(uploaded_file)
    if extracted_text:
        if chat_name not in st.session_state["chats"]:
            st.session_state["chats"][chat_name] = {"messages": [], "context_docs": []}
        st.sidebar.success("✅ Document added to chatbot knowledge!")

# =========================== RETRIEVAL-AUGMENTED GENERATION (RAG) =========================== #
retriever = None
if chat_name not in st.session_state["chats"]:
    st.session_state["chats"][chat_name] = {"messages": [], "context_docs": []}
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.create_documents(chat_session["context_docs"])
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_documents(docs, embeddings)
        retriever = vector_store.as_retriever()
    except Exception as e:
        st.sidebar.error(f"❌ Error setting up RAG: {str(e)}")

# =========================== AI RESPONSE IMPROVEMENT =========================== #
def generate_response(prompt):
    # Feedback-based response adaptation
    cursor.execute("SELECT COUNT(*) FROM chat_history WHERE feedback = 1")
    positive_feedback = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM chat_history WHERE feedback = 0")
    negative_feedback = cursor.fetchone()[0]
    
    adaptive_prompt = "Use a friendly tone." if positive_feedback > negative_feedback else "Provide more in-depth analysis."
    structured_prompt = f"""
    You are a professional AI assistant. Please provide detailed, well-explained responses.
    User input: {prompt}
    """
    try:
        response = llm.invoke(adaptive_prompt + "\n" + structured_prompt)
        return response.content if response else "I couldn't generate a response."
    except Exception as e:
        st.error(f"❌ AI Error: {str(e)}")
        return "I encountered an error while generating a response."

# =========================== DISPLAY CHAT MESSAGES =========================== #
st.subheader(f"💬 {chat_name}")
for i, message in enumerate(messages):
    role = message["role"]
    content = message["content"]
    with st.chat_message(role):
        col1, col2 = st.columns([0.9, 0.1])
        col1.markdown(content)
        if col2.button("📝", key=f"edit_{i}"):
            new_content = st.text_area(f"Edit message {i}", content)
            if st.button("✅ Save", key=f"save_{i}"):
                messages[i]["content"] = new_content
                st.rerun()
        if col2.button("❌", key=f"delete_{i}"):
            del messages[i]
            st.rerun()

# =========================== USER INPUT HANDLING =========================== #
if prompt := st.chat_input("Ask me anything..."):
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response_text = ""
    if retriever:
        try:
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            retrieval_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
            response_text = retrieval_chain.run(prompt)
        except Exception as e:
            st.error(f"❌ Error processing response: {str(e)}")
            response_text = "I encountered an error while generating a response."
    else:
        response_text = generate_response(prompt)
    
    with st.chat_message("assistant"):
        st.markdown(response_text)
    
    messages.append({"role": "assistant", "content": response_text})
    cursor.execute("INSERT INTO chat_history (user_input, ai_response, feedback) VALUES (?, ?, ?)", (prompt, response_text, None))
    conn.commit()

# =========================== CHAT DELETION =========================== #
if st.sidebar.button("🗑️ Delete Chat"):
    del st.session_state["chats"][chat_name]
    st.rerun()
