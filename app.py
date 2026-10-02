import streamlit as st
import os
import tempfile
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool
from langchain_classic.agents import initialize_agent, AgentType

st.title("AI Assistant: DSA, Web Search & File Q&A")

# 1. Sidebar Setup for API Key and File Upload
api_key = st.sidebar.text_input("Enter Google Gemini API Key", type="password")
os.environ["GOOGLE_API_KEY"] = api_key

uploaded_file = st.sidebar.file_uploader("Upload a Text File (.txt)", type=["txt"])

# 2. Initialize LLM & Agent Tools
if api_key:
    # Use Gemini 2.5 Flash for fast reasoning, DSA logic, and tool routing
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
    search_tool.description = "Use this to search the internet for external website information or current events."
    
    tools = [search_tool]
    
    # Tool B: Document Search (Only activates if a file is uploaded)
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tf:
            tf.write(uploaded_file.getbuffer())
            temp_path = tf.name
            
        # Chunk the text into readable pieces
        loader = TextLoader(temp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        
        # Create a local vector database for semantic search
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_db = FAISS.from_documents(chunks, embeddings)
        retriever = vector_db.as_retriever()
        
        # Give the agent access to the database
        doc_tool = create_retriever_tool(
            retriever,
            "Document_Search",
            "Use this tool to search for information inside the uploaded text file."
        )
        tools.append(doc_tool)
        st.sidebar.success("File processed! The AI can now search your document.")

    # Create the Agent that decides which tool to use
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

    # 3. Chat Interface Setup
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # 4. Handle User Input
    user_input = st.chat_input("Ask a DSA question, search the web, or query your file...")
    if user_input:
        st.chat_message("user").write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner("Thinking and routing tools..."):
            response = agent.run(user_input)
            
        st.chat_message("assistant").write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.warning("Please enter your Gemini API Key in the sidebar to start.")