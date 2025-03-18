import os
import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema.runnable import RunnableLambda

# Set up Streamlit app
st.set_page_config(page_title="Interactive AI Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 AI Chatbot with Dynamic User Context")

# Fetch API key with fallback option
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = st.sidebar.text_input("Enter Google AI API Key:", type="password")
    if not GOOGLE_API_KEY:
        st.sidebar.error("API key required!")
        st.stop()

# Configure Google Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the latest Gemini AI model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "context_docs" not in st.session_state:
    st.session_state.context_docs = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Sidebar for context management
st.sidebar.header("📚 Add Context for Better Responses")
context_input = st.sidebar.text_area("Enter context (e.g., company policies, product details, etc.)")

if st.sidebar.button("Add to Knowledge Base"):
    if context_input:
        st.session_state.context_docs.append(context_input)
        st.sidebar.success("Context added successfully!")
        st.rerun()

# Display added contexts with removal option
st.sidebar.subheader("📚 Added Contexts")
for i, doc in enumerate(st.session_state.context_docs):
    st.sidebar.text(f"{i+1}. {doc[:50]}...")
    if st.sidebar.button(f"Remove {i+1}", key=f"remove_{i}"):
        del st.session_state.context_docs[i]
        st.sidebar.success("Context removed!")
        st.rerun()

# Process and store vector embeddings if context exists
if st.session_state.context_docs:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents(st.session_state.context_docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if st.session_state.vector_store is None:
        st.session_state.vector_store = FAISS.from_documents(docs, embeddings)
    else:
        st.session_state.vector_store.add_documents(docs)

    retriever = st.session_state.vector_store.as_retriever()
else:
    retriever = None  # No context provided

# Chat input handling
if prompt := st.chat_input("Ask me anything..."):
    prompt = prompt.strip()
    if not prompt:
        st.warning("Please enter a valid question!")
        st.stop()

    # Store and display user input
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if retriever:
        retrieval_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=st.session_state.memory)

        def process_input(input_text):
            return retrieval_chain.invoke({"question": input_text})

        rag_pipeline = RunnableLambda(process_input)

        # Stream response
        with st.chat_message("assistant"):
            response_container = st.empty()
            response_text = ""

            for chunk in rag_pipeline.stream(prompt):
                response_chunk = chunk.get("answer", "")
                response_text += response_chunk
                response_container.markdown(response_text)  # Update dynamically
    else:
        # No context, use Gemini AI directly
        response = llm.invoke(prompt)
        response_text = response.content if response else "I couldn't generate a response."

        with st.chat_message("assistant"):
            st.markdown(response_text)

    # Store AI response
    st.session_state.messages.append({"role": "assistant", "content": response_text})

# Chat history clear option
if st.sidebar.button("🗑 Clear Chat History"):
    st.session_state.messages = []
    st.session_state.memory.clear()
    st.sidebar.success("Chat history cleared!")
    st.rerun()
