import streamlit as st
import os
import io
import json
from dotenv import load_dotenv

# Core Logic imports
from connectors.gdrive import GDriveConnector
from processing.parser import DocumentParser
from processing.chunker import DocumentChunker
from search.vector_store import VectorStore
from api.llm_service import LLMService

load_dotenv()

# Streamlit Configuration
st.set_page_config(page_title="GDrive AI Assistant", page_icon="🧠", layout="wide")

# Check for required Secrets
def check_secrets():
    missing = []
    if "GEMINI_API_KEY" not in st.secrets and not os.getenv("GEMINI_API_KEY"):
        missing.append("GEMINI_API_KEY")
    if "gdrive_service_account" not in st.secrets:
        missing.append("gdrive_service_account (JSON content)")
    return missing

missing_secrets = check_secrets()
if missing_secrets:
    st.error(f"❌ Missing Secrets in Streamlit Cloud: {', '.join(missing_secrets)}")
    st.info("Please add these to your Streamlit App settings > Secrets.")
    st.stop()

# Initialize Services (Cached to avoid re-loading models)
@st.cache_resource
def get_services():
    # Force the API key from secrets into the environment for LLMService
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
    
    connector = GDriveConnector()
    vector_store = VectorStore()
    llm_service = LLMService()
    parser = DocumentParser()
    chunker = DocumentChunker()
    return connector, vector_store, llm_service, parser, chunker

connector, vector_store, llm_service, parser, chunker = get_services()

# Sidebar for Sync
with st.sidebar:
    st.title("🧠 Drive Settings")
    if st.button("🔄 Sync with Google Drive"):
        with st.status("Syncing Documents...", expanded=True) as status:
            try:
                st.write("Fetching file list...")
                files = connector.list_files()
                
                new_files_indexed = 0
                for file in files:
                    # Check if file needs syncing
                    if vector_store.should_sync(file['id'], file['modifiedTime']):
                        st.write(f"📥 Processing: {file['name']}")
                        content = connector.download_file(file['id'], file['mimeType'])
                        text = parser.parse(content, file['mimeType'])
                        chunks = chunker.chunk(text, metadata={
                            "doc_id": file['id'],
                            "file_name": file['name'],
                            "source": "gdrive"
                        })
                        vector_store.add_chunks(chunks)
                        vector_store.mark_synced(file['id'], file['modifiedTime'])
                        new_files_indexed += 1
                
                status.update(label=f"✅ Sync Complete! ({new_files_indexed} files updated)", state="complete")
            except Exception as e:
                st.error(f"Sync failed: {str(e)}")
    
    st.markdown("---")
    st.metric("Total Knowledge Chunks", len(vector_store.chunks))
    st.metric("Files in Index", len(vector_store.synced_files))

# Main Chat Interface
st.title("Drive Intelligence Assistant")
st.caption("Grounded GDrive RAG • Powered by Gemini 1.5 Flash")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your documents..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            # 1. Retrieval
            relevant_chunks = vector_store.search(prompt, k=5)
            
            # 2. Generation
            answer = llm_service.generate_answer(prompt, relevant_chunks)
            
            # 3. UI Response
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
