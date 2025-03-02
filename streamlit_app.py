import os
import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema.runnable import RunnableLambda
from langchain.schema.output_parser import StrOutputParser

# Set up Streamlit app
st.set_page_config(page_title="Interactive AI Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 AI Chatbot with Dynamic User Context")

# Fetch API key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Google AI API key not found! Set it as an environment variable.")
    st.stop()

# Configure Google Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the latest Gemini AI model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)

# Initialize session state for chat and user context
if "messages" not in st.session_state:
    st.session_state.messages = []
if "context_docs" not in st.session_state:
    st.session_state.context_docs = []

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Section to accept additional user-provided knowledge/context
st.sidebar.header("📚 Add Context for Better Responses")
context_input = st.sidebar.text_area("Enter context (e.g., company policies, product details, etc.)")

if st.sidebar.button("Add to Knowledge Base"):
    if context_input:
        st.session_state.context_docs.append(context_input)
        st.sidebar.success("Context added successfully!")

# Process user-provided context into a retrievable format
if st.session_state.context_docs:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents(st.session_state.context_docs)

    # Generate embeddings & create vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(docs, embeddings)

    # Initialize retriever
    retriever = vector_store.as_retriever()
else:
    retriever = None  # No user context provided

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Store and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Use RAG if context is provided, else rely on Gemini AI only
    if retriever:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        retrieval_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

        def process_input(input_text):
            return retrieval_chain.invoke(input_text)

        rag_pipeline = (
            RunnableLambda(process_input)  # Process input using RAG
            | StrOutputParser()  # Ensure output is a clean string
        )

        # Stream response
        with st.chat_message("assistant"):
            response_container = st.empty()
            response_text = ""

            for chunk in rag_pipeline.stream(prompt):
                response_text += chunk
                response_container.markdown(response_text)

    else:
        # No context, use Gemini AI directly
        response = llm.invoke(prompt)
        response_text = response.content if response else "I couldn't generate a response."

        # Display response
        with st.chat_message("assistant"):
            st.markdown(response_text)

    # Store AI response
    st.session_state.messages.append({"role": "assistant", "content": response_text})
