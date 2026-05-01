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

st.set_page_config(page_title="GDrive AI Assistant", page_icon="🧠", layout="wide")

# Session State for Credentials
if "auth_ready" not in st.session_state:
    st.session_state.auth_ready = False

# --- SETUP SCREEN ---
if not st.session_state.auth_ready:
    st.title("🔐 GDrive RAG Setup")
    st.info("To protect your credentials, please provide them for this session. They are not stored permanently.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Auto-fill from env if available
        default_key = os.getenv("GEMINI_API_KEY", "")
        gemini_key = st.text_input("Enter Gemini API Key", value=default_key, type="password")
        st.markdown("[Get a key here](https://aistudio.google.com/app/apikey)")
        
    with col2:
        uploaded_file = st.file_uploader("Upload service_account.json", type="json")
        local_creds_path = "data/credentials/service_account.json"
        has_local_creds = os.path.exists(local_creds_path)
        if has_local_creds:
            st.success("✅ Local service_account.json found!")
    
    if st.button("🚀 Initialize System"):
        # Check if we use uploaded or local
        final_creds_ready = False
        if uploaded_file:
            os.makedirs("data/credentials", exist_ok=True)
            with open(local_creds_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            final_creds_ready = True
        elif has_local_creds:
            final_creds_ready = True
            
        if gemini_key and final_creds_ready:
            try:
                # 1. Set Gemini Key
                os.environ["GEMINI_API_KEY"] = gemini_key
                
                # 2. Save service account temporarily
                os.makedirs("data/credentials", exist_ok=True)
                with open("data/credentials/service_account.json", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # 3. Test initialization
                with st.spinner("Loading models..."):
                    st.session_state.connector = GDriveConnector()
                    st.session_state.vector_store = VectorStore()
                    st.session_state.llm_service = LLMService()
                    st.session_state.parser = DocumentParser()
                    st.session_state.chunker = DocumentChunker()
                    st.session_state.auth_ready = True
                    st.rerun()
            except Exception as e:
                st.error(f"Initialization Failed: {e}")
        else:
            st.warning("Please provide both the API Key and the JSON file.")
    st.stop()

# --- MAIN APP SCREEN ---
connector = st.session_state.connector
vector_store = st.session_state.vector_store
llm_service = st.session_state.llm_service
parser = st.session_state.parser
chunker = st.session_state.chunker

# Sidebar
with st.sidebar:
    st.title("🧠 Drive Settings")
    if st.button("🔄 Sync with Google Drive"):
        with st.status("Syncing Documents...", expanded=True) as status:
            try:
                files = connector.list_files()
                new_files_indexed = 0
                for file in files:
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
                status.update(label=f"✅ Sync Complete! ({new_files_indexed} new files)", state="complete")
            except Exception as e:
                st.error(f"Sync failed: {e}")
    
    st.markdown("---")
    st.metric("Knowledge Chunks", len(vector_store.chunks))
    if st.button("🗑️ Clear Credentials"):
        st.session_state.auth_ready = False
        st.rerun()

# Chat UI
st.title("Drive Intelligence Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your documents..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            relevant_chunks = vector_store.search(prompt, k=5)
            answer = llm_service.generate_answer(prompt, relevant_chunks)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
