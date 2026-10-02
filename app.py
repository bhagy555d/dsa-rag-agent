import os
import tempfile
import time
import streamlit as st
from langchain_classic.agents import AgentType, initialize_agent
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.title("AI Assistant: DSA, Web Search & File Q&A")

# --- Step 1: Sidebar & File Upload Inputs (Define variables first) ---
api_key = st.sidebar.text_input("Enter Google Gemini API Key", type="password")
uploaded_file = st.file_uploader(
    "Upload a document", 
    type=["txt", "pdf", "csv"]
)

# --- Step 2: Main Logic ---
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=api_key
    )
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        api_key=api_key
    )

    search_tool = DuckDuckGoSearchRun()
    search_tool.name = "Web_Search"
    search_tool.description = "Search the internet for external website information or current events."
    tools = [search_tool]

    # Now uploaded_file is guaranteed to exist
    if uploaded_file is not None:
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()

        if "vector_db" not in st.session_state or st.session_state.get("last_uploaded") != uploaded_file.name:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tf:
                tf.write(uploaded_file.getbuffer())
                temp_path = tf.name

            if file_ext == ".pdf":
                loader = PyPDFLoader(temp_path)
            elif file_ext == ".csv":
                loader = CSVLoader(temp_path)
            else:
                loader = TextLoader(temp_path, encoding="utf-8")

            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_documents(docs)

            st.session_state.vector_db = FAISS.from_documents(chunks, embeddings)
            st.session_state.last_uploaded = uploaded_file.name
            st.sidebar.success(f"{uploaded_file.name} processed successfully!")

        retriever = st.session_state.vector_db.as_retriever()
        doc_tool = create_retriever_tool(
            retriever,
            "Document_Search",
            "Use this tool to search for specific content inside the uploaded document."
        )
        tools.append(doc_tool)