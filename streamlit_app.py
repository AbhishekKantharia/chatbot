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
import speech_recognition as sr
from gtts import gTTS
import matplotlib.pyplot as plt
import pdfkit

# Configure Streamlit
st.set_page_config(page_title="Smart AI Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Smart AI Chatbot")

# Fetch Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Google API key is missing! Set it as an environment variable.")
    st.stop()

# Configure Google Gemini AI
genai.configure(api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)

# Initialize session state for multi-user chat
user_id = st.query_params.get("user", "default")
if user_id not in st.session_state:
    st.session_state[user_id] = {"messages": [], "context_docs": []}

messages = st.session_state[user_id]["messages"]

# Sidebar: Document Upload for RAG
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
        st.session_state[user_id]["context_docs"].append(extracted_text)
        st.sidebar.success("✅ Document added to chatbot knowledge!")

# Process user-provided context for Retrieval-Augmented Generation (RAG)
retriever = None
if st.session_state[user_id]["context_docs"]:
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.create_documents(st.session_state[user_id]["context_docs"])
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_documents(docs, embeddings)
        retriever = vector_store.as_retriever()
    except Exception as e:
        st.sidebar.error(f"❌ Error setting up RAG: {str(e)}")

# Thematic chatbot selection
theme = st.sidebar.selectbox("🎨 Choose Chatbot Theme", ["Default", "Business", "Casual", "Legal"])

def generate_response(prompt, theme):
    if theme == "Business":
        return f"📊 Professional Response: {prompt}"
    elif theme == "Casual":
        return f"😎 Chill Response: {prompt}"
    elif theme == "Legal":
        return f"⚖️ Legal Analysis: {prompt}"
    return f"🤖 Default: {prompt}"

# Chat Input
if prompt := st.chat_input("Ask me anything..."):
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response_text = ""  # Ensure response_text is initialized

    # Use RAG if context is available, else only AI
    if retriever:
        try:
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            retrieval_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

            def process_input(input_text):
                return retrieval_chain.invoke(input_text)

            rag_pipeline = RunnableLambda(process_input)

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

        except Exception as e:
            st.error(f"❌ Error processing response: {str(e)}")
            response_text = "I encountered an error while generating a response."

    else:
        try:
            response = llm.invoke(prompt)
            response_text = response.content if response else "I couldn't generate a response."

            with st.chat_message("assistant"):
                st.markdown(response_text)
        except Exception as e:
            st.error(f"❌ AI Error: {str(e)}")
            response_text = "I encountered an error while generating a response."

    # Store assistant response
    messages.append({"role": "assistant", "content": response_text})

    # Generate improved voice response
    try:
        tts = gTTS(response_text, lang="en")
        tts.save("response.mp3")
        st.audio("response.mp3")
    except Exception as e:
        st.error(f"❌ Audio generation error: {str(e)}")

# Chat History Export
if st.sidebar.button("📄 Download Chat as PDF"):
    chat_history = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
    
    if chat_history.strip():
        try:
            pdfkit.from_string(chat_history, "chat.pdf")
            with open("chat.pdf", "rb") as file:
                st.sidebar.download_button("Download PDF", file, file_name="chat_history.pdf")
        except Exception as e:
            st.sidebar.error(f"❌ PDF generation error: {str(e)}")
    else:
        st.sidebar.warning("⚠️ Chat history is empty! No PDF generated.")

# Admin Dashboard (Analytics)
st.sidebar.header("📊 Chatbot Analytics")
st.sidebar.metric("Total Chats", str(len(messages)))
st.sidebar.metric("Unique Users", "1")  # Replace with dynamic user tracking

fig, ax = plt.subplots()
ax.bar(["Positive", "Neutral", "Negative"], [60, 30, 10])  # Fake sentiment analysis
st.sidebar.pyplot(fig)
