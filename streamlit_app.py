import os
import streamlit as st
import firebase_admin
from firebase_admin import auth, credentials, firestore
import google.generativeai as genai
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import docx2txt
from gtts import gTTS
import pdfkit
import socket
import datetime
import matplotlib.pyplot as plt
import fastapi
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

# 🎯 **Firebase Authentication Setup**
cred = credentials.Certificate("firebase_credentials.json")  # Ensure this is set up in Firebase
firebase_admin.initialize_app(cred)
db = firestore.client()

# 🎯 **FastAPI Backend**
app = fastapi.FastAPI()

# Allow CORS (for frontend to communicate with backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📌 **Streamlit UI Configuration**
st.set_page_config(page_title="Smart AI Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Smart AI Chatbot")

# 🎯 **Authentication Functions**
def authenticate_user(email, password):
    try:
        user = auth.get_user_by_email(email)
        return user.uid
    except:
        return None

def register_user(email, password):
    try:
        user = auth.create_user(email=email, password=password)
        return user.uid
    except Exception as e:
        st.error(f"Registration failed: {e}")
        return None

# 🎯 **User Login**
if "user" not in st.session_state:
    st.session_state["user"] = None

if not st.session_state["user"]:
    st.sidebar.header("🔐 Login / Register")
    auth_option = st.sidebar.radio("Select Option", ["Login", "Register"])
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type="password")
    
    if auth_option == "Login":
        if st.sidebar.button("Login"):
            user_id = authenticate_user(email, password)
            if user_id:
                st.session_state["user"] = user_id
                st.experimental_rerun()
            else:
                st.sidebar.error("Invalid login credentials.")

    elif auth_option == "Register":
        if st.sidebar.button("Register"):
            user_id = register_user(email, password)
            if user_id:
                st.session_state["user"] = user_id
                st.sidebar.success("Account created! Please log in.")
                st.experimental_rerun()
else:
    st.sidebar.success(f"Logged in as {st.session_state['user']}")
    if st.sidebar.button("Logout"):
        st.session_state["user"] = None
        st.experimental_rerun()

# 🎯 **Google AI Configuration**
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)

# 🎯 **Analytics Tracking**
def log_chat(user_id, user_input, bot_response):
    doc_ref = db.collection("chats").document(user_id).collection("messages").document()
    doc_ref.set({
        "timestamp": datetime.datetime.now(),
        "user_input": user_input,
        "bot_response": bot_response
    })

# 🎯 **Retrieve Chat History**
def get_chat_history(user_id):
    chat_ref = db.collection("chats").document(user_id).collection("messages")
    chat_history = chat_ref.order_by("timestamp").stream()
    return [{"role": "user", "content": doc.to_dict()["user_input"]} for doc in chat_history] + \
           [{"role": "assistant", "content": doc.to_dict()["bot_response"]} for doc in chat_history]

# 🎯 **Sentiment Analysis (Basic)**
def analyze_sentiment(text):
    if "good" in text or "happy" in text:
        return "Positive"
    elif "bad" in text or "sad" in text:
        return "Negative"
    else:
        return "Neutral"

# 🎯 **IP Address Banning**
def get_user_ip():
    try:
        return socket.gethostbyname(socket.gethostname())
    except:
        return "Unknown"

BANNED_IPS = ["192.168.1.100", "203.0.113.45"]
user_ip = get_user_ip()
if user_ip in BANNED_IPS:
    st.error("🚫 You have been banned from using this chatbot.")
    st.stop()

# 🎯 **Chat Interface**
messages = get_chat_history(st.session_state["user"]) if st.session_state["user"] else []
st.subheader("💬 Chat")

for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything..."):
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response_text = ""
    response = llm.invoke(prompt)
    response_text = response.content if response else "I couldn't generate a response."

    with st.chat_message("assistant"):
        st.markdown(response_text)

    # Store chat in Firebase
    log_chat(st.session_state["user"], prompt, response_text)

    # 🎯 **Audio Response**
    try:
        tts = gTTS(response_text, lang="en")
        tts.save("response.mp3")
        st.audio("response.mp3")
    except Exception as e:
        st.error(f"❌ Audio generation error: {str(e)}")

# 🎯 **Advanced Analytics**
st.sidebar.header("📊 Analytics")
total_chats = len(messages)
positive_count = sum(1 for m in messages if analyze_sentiment(m["content"]) == "Positive")
negative_count = sum(1 for m in messages if analyze_sentiment(m["content"]) == "Negative")
neutral_count = total_chats - (positive_count + negative_count)

st.sidebar.metric("Total Chats", total_chats)
st.sidebar.metric("Positive Responses", positive_count)
st.sidebar.metric("Negative Responses", negative_count)

# 🎯 **Graph Visualization**
fig, ax = plt.subplots()
ax.bar(["Positive", "Neutral", "Negative"], [positive_count, neutral_count, negative_count])
st.sidebar.pyplot(fig)

# 🎯 **Deployment with FastAPI**
@app.get("/")
def home():
    return JSONResponse(content={"message": "AI Chatbot API is running!"})

@app.get("/health")
def health_check():
    return JSONResponse(content={"status": "OK"})

@app.post("/chat")
async def chat_api(data: dict):
    user_input = data.get("user_input", "")
    response = llm.invoke(user_input)
    return JSONResponse(content={"bot_response": response.content if response else "Error generating response."})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
