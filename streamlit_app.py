import streamlit as st
import google.generativeai as genai
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage
from langchain.vectorstores import FAISS
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.pydantic_v1 import BaseModel
from langchain.schema.runnable import RunnableLambda
from langchain.schema.output_parser import StrOutputParser

# Set up Streamlit app
st.set_page_config(page_title="AI Chatbot with RAG", page_icon="🤖", layout="wide")
st.title("🤖 AI Chatbot with Gemini 1.5, RAG & LangChain")

# Fetch API key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Google AI API key not found! Set it as an environment variable.")
    st.stop()

# Configure Google Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize latest Gemini AI model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)

# -------------------- RAG: Load & Process Documents --------------------

# Load documents
loader = TextLoader("knowledge_base.txt")  # Custom knowledge source
documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Create embeddings & vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = FAISS.from_documents(docs, embeddings)

# Initialize retriever
retriever = vector_store.as_retriever()

# -------------------- Memory & Conversational RAG Chain --------------------

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

retrieval_chain = ConversationalRetrievalChain.from_llm(
    llm=llm, retriever=retriever, memory=memory
)

# -------------------- AI Model Schema with Pydantic --------------------

class UserQuery(BaseModel):
    question: str

class AIResponse(BaseModel):
    answer: str

# -------------------- Streamlit Chat UI --------------------

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):

    # Store user input
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Define a Runnable pipeline with RAG
    def process_input(input_text):
        structured_input = UserQuery(question=input_text)
        return retrieval_chain.invoke(structured_input.question)

    # Create a Runnable object
    rag_pipeline = (
        RunnableLambda(process_input)  # Run the RAG pipeline
        | StrOutputParser()  # Extract response as a string
    )

    # Stream response in real-time
    with st.chat_message("assistant"):
        response_container = st.empty()
        response_text = ""

        for chunk in rag_pipeline.stream(prompt):
            response_text += chunk
            response_container.markdown(response_text)  # Update progressively

        # Store AI response
        ai_response = AIResponse(answer=response_text)
        st.session_state.messages.append({"role": "assistant", "content": ai_response.answer})
