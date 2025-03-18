import os
import streamlit as st
import google.generativeai as genai
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import docx2txt
import speech_recognition as sr
from gtts import gTTS
import pdfkit
from pymongo import MongoClient
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import threading

# FastAPI instance
app = FastAPI()

# MongoDB Connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
db = client["chatbot_db"]
chat_collection = db["chats"]

# Streamlit UI Configuration
st.set_page_config(page_title="Smart AI Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Smart AI Chatbot with API & Database")

# Fetch Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Google API key is missing! Set it as an environment variable.")
    st.stop()

# Configure Google Gemini AI
genai.configure(api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)

# Multi-user support
user_id = st.experimental_get_query_params().get("user", ["default"])[0]
if user_id not in st.session_state:
    st.session_state[user_id] = {"messages": [], "context_docs": []}

messages = st.session_state[user_id]["messages"]

# Sidebar options - File Upload for RAG
st.sidebar.header("📂 Upload Documents for RAG")
uploaded_file = st.sidebar.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])

if uploaded_file:
    def extract_text(file):
        if file.name.endswith(".pdf"):
            reader = PdfReader(file)
            return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif file.name.endswith(".docx"):
            return docx2txt.process(file)
        else:
            return file.read().decode("utf-8")

    extracted_text = extract_text(uploaded_file)
    st.session_state[user_id]["context_docs"].append(extracted_text)
    st.sidebar.success("Document added to chatbot knowledge!")

# Process user-provided context into a retrievable format
if st.session_state[user_id]["context_docs"]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents(st.session_state[user_id]["context_docs"])
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever()
else:
    retriever = None

# Voice Input & Output
st.sidebar.header("🎤 Voice Input & Output")
if st.sidebar.button("🎙️ Speak"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.sidebar.info("Listening...")
        audio = recognizer.listen(source)
    try:
        voice_prompt = recognizer.recognize_google(audio)
        st.sidebar.write(f"**You said:** {voice_prompt}")
    except:
        st.sidebar.error("Could not recognize speech!")

# Chat Input
if prompt := st.chat_input("Ask me anything..."):
    messages.append({"role": "user", "content": prompt})
    chat_collection.insert_one({"user_id": user_id, "role": "user", "content": prompt})  # Store in DB

    with st.chat_message("user"):
        st.markdown(prompt)

    # Use RAG if context exists, else direct AI
    if retriever:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        retrieval_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

        def process_input(input_text):
            return retrieval_chain.invoke(input_text)

        rag_pipeline = RunnableLambda(process_input)

        response_text = ""
        with st.chat_message("assistant"):
            response_container = st.empty()

            for chunk in rag_pipeline.stream(prompt):
                if isinstance(chunk, str):
                    response_text += chunk
                elif hasattr(chunk, "text"):
                    response_text += chunk.text
                elif hasattr(chunk, "content"):
                    response_text += chunk.content

            response_container.markdown(response_text)

    else:
        response = llm.invoke(prompt)
        response_text = response.content if response else "I couldn't generate a response."

        with st.chat_message("assistant"):
            st.markdown(response_text)

    # Store assistant response
    messages.append({"role": "assistant", "content": response_text})
    chat_collection.insert_one({"user_id": user_id, "role": "assistant", "content": response_text})

    # Generate voice response
    tts = gTTS(response_text)
    tts.save("response.mp3")
    st.audio("response.mp3")

# Chat History Export
if st.sidebar.button("📄 Download Chat as PDF"):
    chat_history = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
    pdfkit.from_string(chat_history, "chat.pdf")
    with open("chat.pdf", "rb") as file:
        st.sidebar.download_button("Download PDF", file, file_name="chat_history.pdf")

# FastAPI Models
class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    response: str

# FastAPI Endpoints
@app.post("/api/chat", response_model=ChatResponse)
def chat_api(request: ChatRequest):
    response = llm.invoke(request.message)
    chat_response = response.content if response else "I couldn't generate a response."

    # Store in MongoDB
    chat_collection.insert_one({"user_id": request.user_id, "role": "user", "content": request.message})
    chat_collection.insert_one({"user_id": request.user_id, "role": "assistant", "content": chat_response})

    return {"response": chat_response}

@app.get("/api/chat/history/{user_id}")
def get_chat_history(user_id: str):
    history = list(chat_collection.find({"user_id": user_id}, {"_id": 0}))
    return {"history": history}

# Run FastAPI server in a separate thread
def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

threading.Thread(target=run_fastapi, daemon=True).start()
