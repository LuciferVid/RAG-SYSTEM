import streamlit as st
import os
import time
from dotenv import load_dotenv

# Import our core logic (now integrated)
from connectors.gdrive import GDriveConnector
from processing.parser import DocumentParser
from processing.chunker import DocumentChunker
from search.vector_store import VectorStore
from api.llm_service import LLMService

load_dotenv()

# Streamlit Page Config
st.set_page_config(
    page_title="GDrive Intelligence RAG",
    page_icon="🧠",
    layout="wide"
)

# Initialize Services
@st.cache_resource
def init_services():
    # Use secrets if available (for Streamlit Cloud), else env vars
    gemini_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key
        
    connector = GDriveConnector()
    vector_store = VectorStore()
    llm_service = LLMService()
    return connector, vector_store, llm_service

connector, vector_store, llm_service = init_services()

# UI Styling
st.markdown("""
    <style>
    .source-tag {
        background-color: #1e2129;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8em;
        margin-right: 5px;
        border: 1px solid #3d444d;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🧠 GDrive RAG")
    st.markdown("---")
    
    st.subheader("System Control")
    if st.button("🔄 Sync Google Drive"):
        with st.spinner("Fetching and indexing documents..."):
            try:
                files = connector.list_files()
                parser = DocumentParser()
                chunker = DocumentChunker()
                
                new_chunks_count = 0
                for file in files:
                    if not vector_store.should_sync(file['id'], file['modifiedTime']):
                        continue
                        
                    content = connector.download_file(file['id'], file['mimeType'])
                    if content:
                        text = parser.parse(content, file['mimeType'])
                        chunks = chunker.chunk(text, metadata={
                            "doc_id": file['id'],
                            "file_name": file['name'],
                            "source": "gdrive"
                        })
                        vector_store.add_chunks(chunks)
                        vector_store.mark_synced(file['id'], file['modifiedTime'])
                        new_chunks_count += len(chunks)
                
                if new_chunks_count > 0:
                    st.success(f"Sync complete! Added {new_chunks_count} new chunks.")
                else:
                    st.info("Everything is already up to date.")
            except Exception as e:
                st.error(f"Sync Error: {e}")
    
    st.markdown("---")
    st.subheader("Knowledge Base")
    st.metric("Total Chunks", len(vector_store.chunks))
    st.metric("Files Synced", len(vector_store.synced_files))

# Main Interface
st.title("Drive Intelligence Assistant")
st.info("Ask anything about your shared Google Drive documents.")

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            st.markdown("---")
            for source in message["sources"]:
                st.markdown(f'<span class="source-tag">📄 {source}</span>', unsafe_allow_html=True)

# User Input
if prompt := st.chat_input("What would you like to know?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Searching and thinking..."):
            # RAG Pipeline
            results = vector_store.search(prompt, k=5)
            answer = llm_service.generate_answer(prompt, results)
            
            # Extract unique sources
            sources = list(set([r['metadata']['file_name'] for r in results]))
            
            st.markdown(answer)
            if sources:
                st.markdown("---")
                for source in sources:
                    st.markdown(f'<span class="source-tag">📄 {source}</span>', unsafe_allow_html=True)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "sources": sources
            })
