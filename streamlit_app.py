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
import socket
import requests

# ======================== CONFIGURE STREAMLIT ========================
st.set_page_config(page_title="Smart AI Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 AI Chatbot with Google Gemini & More!")

# Fetch Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Google API key is missing! Set it as an environment variable.")
    st.stop()

# Configure Google Gemini AI
genai.configure(api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)

# ======================== MULTIPLE CHAT SESSIONS ========================
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
    chat_session["context_docs"].append(extracted_text)
    st.sidebar.success("✅ Document added to chatbot knowledge!")

# ======================== DISPLAY CHAT MESSAGES ========================
st.subheader(f"💬 {chat_name}")
for message in messages:
    st.chat_message(message["role"]).markdown(message["content"])

# ======================== GOOGLE GEMINI IMAGE GENERATION ========================
st.sidebar.header("🎨 Generate AI Images")
image_prompt = st.sidebar.text_input("Enter image description")
if st.sidebar.button("🎨 Generate Image"):
    if image_prompt:
        response = genai.generate_content(model="gemini-1.5-vision", contents=[{"role": "user", "parts": [{"text": image_prompt}]}])
        st.image(response["image"], caption="Generated by Google Gemini Vision AI")

# ======================== VOICE INPUT (SPEECH-TO-TEXT) ========================
st.sidebar.header("🎤 Voice Input")
if st.sidebar.button("🎙️ Record Voice"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.sidebar.write("Listening... Speak now!")
        audio = recognizer.listen(source)

    try:
        prompt = recognizer.recognize_google(audio)
        st.sidebar.success(f"Recognized: {prompt}")
    except sr.UnknownValueError:
        st.sidebar.error("Could not understand audio.")
    except sr.RequestError:
        st.sidebar.error("Speech recognition service unavailable.")

# ======================== WEATHER & NEWS UPDATES ========================
st.sidebar.header("🌍 Live News & Weather")

def get_weather():
    api_key = "YOUR_OPENWEATHER_API_KEY"
    city = "New York"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url).json()
    return f"🌡️ {response['main']['temp']}°C | {response['weather'][0]['description']}"

if st.sidebar.button("🌦️ Get Weather"):
    try:
        weather_info = get_weather()
        st.sidebar.success(weather_info)
    except:
        st.sidebar.error("Could not fetch weather data.")

def get_news():
    api_key = "YOUR_NEWSAPI_KEY"  # Replace with a valid API key
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}"
    
    try:
        response = requests.get(url).json()
        if "articles" in response and response["articles"]:
            return response["articles"][0]["title"]
        else:
            return "No news found."
    except Exception as e:
        return f"Error fetching news: {str(e)}"

if st.sidebar.button("📰 Get News"):
    news_info = get_news()
    st.sidebar.success(news_info)

# ======================== CHAT FUNCTIONALITY ========================
prompt = st.chat_input("Ask me anything...")

if prompt:
    messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    response_text = ""

    if retriever:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        retrieval_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
        response_text = retrieval_chain.run(prompt)
    else:
        response_text = llm.invoke(prompt).content

    messages.append({"role": "assistant", "content": response_text})
    st.chat_message("assistant").markdown(response_text)

    # Convert AI response to speech
    tts = gTTS(response_text, lang="en")
    tts.save("response.mp3")
    st.audio("response.mp3")

# ======================== DELETE CHAT ========================
if st.sidebar.button("🗑️ Delete Chat"):
    del st.session_state["chats"][chat_name]
    st.rerun()
