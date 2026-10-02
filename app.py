import os
import tempfile
import streamlit as st
from langchain_classic.agents import AgentType, initialize_agent
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.title("AI Assistant: DSA, Web Search & File Q&A")

# 1. Sidebar Setup
api_key = st.sidebar.text_input("Enter Google Gemini API Key", type="password")
uploaded_file = st.file_uploader(
    "Upload a document", 
    type=["txt", "pdf", "csv"]
)

# 2. Main Logic
if api_key:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=api_key
    )
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        api_key=api_key
    )

    # Tool A: Web Search
    search_tool = DuckDuckGoSearchRun()
    search_tool.name = "Web_Search"
    search_tool.description = "Search the internet for external website information or current events."
    tools = [search_tool]

    # Tool B: Document Search with Session Caching
    if uploaded_file is not None:
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()

        # Cache vector store so it doesn't re-index on every single user prompt
        if "vector_db" not in st.session_state or st.session_state.get("last_uploaded") != uploaded_file.name:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tf:
                tf.write(uploaded_file.getbuffer())
                temp_path = tf.name

            # Route by document format
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

    # Agent Initialization
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

    # Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    user_input = st.chat_input("Ask a DSA question, search the web, or query your file...")
    if user_input:
        st.chat_message("user").write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("Processing..."):
            response = agent.run(user_input)

        st.chat_message("assistant").write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.warning("Please enter your Gemini API Key in the sidebar to start.")