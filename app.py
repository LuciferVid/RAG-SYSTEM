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

# Initialize Services (Cached to avoid re-loading models)
@st.cache_resource
def get_services():
    connector = GDriveConnector()
    vector_store = VectorStore()
    llm_service = LLMService()
    parser = DocumentParser()
    chunker = DocumentChunker()
    return connector, vector_store, llm_service, parser, chunker

try:
    connector, vector_store, llm_service, parser, chunker = get_services()
except Exception as e:
    st.error(f"Failed to initialize services. Ensure credentials are set correctly in the environment. Error: {e}")
    st.stop()

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
