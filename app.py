import os
from langchain_huggingface import HuggingFaceEmbeddings
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

# --- Step 1: Sidebar Setup for API Key and Document Ingestion ---
api_key = st.sidebar.text_input("Enter Google Gemini API Key", type="password")
uploaded_file = st.sidebar.file_uploader(
    "Upload a document (.pdf, .csv, .txt)", 
    type=["txt", "pdf", "csv"]
)

# --- Step 2: Main Application Logic ---
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

    # Initialize the stable 2.5 Flash model
    llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=api_key
)
    # Use local HuggingFace embeddings to completely bypass Google rate limits
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    # Tool A: DuckDuckGo Web Search
    search_tool = DuckDuckGoSearchRun()
    search_tool.name = "Web_Search"
    search_tool.description = "Use this to search the internet for external website information, live data, or current events."
    tools = [search_tool]

    # Tool B: Document Search (Only activated if a file is uploaded)
    if uploaded_file is not None:
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()

        # Cache vector store in session state to prevent re-indexing on every chat input
        if "vector_db" not in st.session_state or st.session_state.get("last_uploaded") != uploaded_file.name:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tf:
                tf.write(uploaded_file.getbuffer())
                temp_path = tf.name

            # Format-specific document loaders
            if file_ext == ".pdf":
                loader = PyPDFLoader(temp_path)
            elif file_ext == ".csv":
                loader = CSVLoader(temp_path)
            else:
                loader = TextLoader(temp_path, encoding="utf-8")

            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_documents(docs)

            if chunks:
                with st.spinner("Processing and indexing document chunks..."):
                    # Initialize FAISS with the first batch
                    st.session_state.vector_db = FAISS.from_documents(chunks[:5], embeddings)

                    # Batch append remaining chunks with a 2-second sleep to respect the 15 RPM rate limit
                    for i in range(5, len(chunks), 5):
                        time.sleep(2)
                        st.session_state.vector_db.add_documents(chunks[i : i + 5])

                st.session_state.last_uploaded = uploaded_file.name
                st.sidebar.success(f"{uploaded_file.name} indexed successfully!")
            else:
                st.sidebar.warning("Uploaded file contains no readable text.")

        if "vector_db" in st.session_state:
            retriever = st.session_state.vector_db.as_retriever()
            doc_tool = create_retriever_tool(
                retriever,
                "Document_Search",
                "Use this tool to search for specific content, facts, or data inside the uploaded document."
            )
            tools.append(doc_tool)

    # Initialize the zero-shot ReAct agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

    # --- Step 3: Chat History Management ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # --- Step 4: User Query Execution ---
    user_input = st.chat_input("Ask a DSA question, search the web, or query your uploaded document...")
    if user_input:
        st.chat_message("user").write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("Thinking and routing query..."):
            time.sleep(1)  # Buffer against API rate spikes
            response = agent.run(user_input)

        st.chat_message("assistant").write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.warning("Please enter your Google Gemini API Key in the sidebar to begin.")