import os
import streamlit as st
import google.generativeai as genai
import openai
import requests
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema.runnable import RunnableLambda
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import docx2txt
import speech_recognition as sr
from gtts import gTTS
import socket

# ======================== CONFIGURE STREAMLIT ========================
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Smart AI Chatbot")

# Fetch Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Google API key is missing! Set it as an environment variable.")
    st.stop()

# Get user IP address for banning feature
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

# Configure Google Gemini AI
genai.configure(api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)

# ======================== PERSONALIZATION ========================
if "username" not in st.session_state:
    st.session_state["username"] = st.text_input("Enter your name:", "")

if st.session_state["username"]:
    st.write(f"👋 Welcome back, {st.session_state['username']}!")

# ======================== AI PERSONALITY SELECTION ========================
personality = st.sidebar.selectbox("🤖 Choose AI Personality", ["Friendly", "Sarcastic", "Professional"])
response_style = {
    "Friendly": "Hey there! 😊 Here's what I think: ",
    "Sarcastic": "Oh wow, what a groundbreaking question... 😏 Here's your answer: ",
    "Professional": "According to my analysis, here’s the best response: "
}[personality]

# ======================== DOCUMENT UPLOAD (RAG) ========================
st.sidebar.header("📂 Upload Documents for RAG")
uploaded_file = st.sidebar.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])

def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif file.name.endswith(".docx"):
        return docx2txt.process(file)
    else:
        return file.read().decode("utf-8")

retriever = None
if uploaded_file:
    extracted_text = extract_text(uploaded_file)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents([extracted_text])
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever()
    st.sidebar.success("✅ Document added to chatbot knowledge!")

# ======================== DISPLAY CHAT MESSAGES ========================
st.subheader("💬 Chat")
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "👤").markdown(message["content"])

# ======================== USER INPUT (TEXT & VOICE) ========================
prompt = st.chat_input("Ask me anything...")

def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("🎙️ Speak now...")
        audio = recognizer.listen(source)
        try:
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand that."

if st.sidebar.button("🎤 Speak Instead"):
    prompt = get_voice_input()
    st.write(f"You said: {prompt}")

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="👤").markdown(prompt)

    response_text = ""

    # Use RAG if context is available, else only AI
    if retriever:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        retrieval_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
        response_text = retrieval_chain.run(prompt)
    else:
        response_text = response_style + llm.invoke(prompt).content

    st.session_state["messages"].append({"role": "assistant", "content": response_text})
    st.chat_message("assistant", avatar="🤖").markdown(response_text)

    # Convert response to speech
    tts = gTTS(response_text, lang="en")
    tts.save("response.mp3")
    st.audio("response.mp3")

# ======================== AI IMAGE GENERATION ========================
def generate_image(prompt):
    response = openai.Image.create(prompt=prompt, n=1, size="1024x1024")
    st.image(response['data'][0]['url'])

if st.sidebar.button("🎨 Generate AI Art"):
    generate_image(prompt)

# ======================== LIVE WEATHER & NEWS ========================
def get_weather(city):
    api_key = "your_openweather_api_key"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url).json()
    return f"🌡️ {response['main']['temp']}°C, {response['weather'][0]['description']}"

if st.sidebar.button("🌍 Get Weather"):
    st.write(get_weather("New York"))

def get_news():
    url = "https://newsapi.org/v2/top-headlines?country=us&apiKey=your_newsapi_key"
    response = requests.get(url).json()
    return response["articles"][0]["title"]

if st.sidebar.button("📰 Get Latest News"):
    st.write(get_news())

# ======================== CHAT STATS & DELETE CHAT ========================
st.sidebar.subheader("📊 Chat Stats")
st.sidebar.write(f"Total Messages: {len(st.session_state['messages'])}")

if st.sidebar.button("🗑️ Delete Chat"):
    st.session_state["messages"] = []
    st.rerun()
