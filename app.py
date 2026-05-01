import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Prioritize the Cloud API URL (e.g., from ngrok), then local
API_URL = st.secrets.get("API_URL") or os.getenv("API_URL", "http://localhost:8002")

st.set_page_config(
    page_title="GDrive Intelligence RAG",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for UI
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
    st.info(f"Connected to: {API_URL}")
    st.markdown("---")
    
    st.subheader("System Control")
    if st.button("🔄 Sync Google Drive"):
        with st.spinner("Requesting sync from backend..."):
            try:
                response = requests.post(f"{API_URL}/sync-drive")
                if response.status_code == 200:
                    st.success("Sync started!")
                else:
                    st.error("Failed to trigger sync.")
            except Exception as e:
                st.error(f"Error: {e}")
    
    st.markdown("---")
    st.subheader("Knowledge Base")
    try:
        status = requests.get(f"{API_URL}/status").json()
        st.metric("Indexed Chunks", status.get("indexed_chunks", 0))
        st.metric("Synced Files", status.get("synced_files_count", 0))
    except:
        st.warning("Could not reach backend. Is ngrok running?")

# Main Interface
st.title("Drive Intelligence Assistant")
st.markdown("Ask anything about your shared Google Drive documents.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
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
        with st.spinner("Consulting knowledge base..."):
            try:
                response = requests.post(f"{API_URL}/ask", json={"query": prompt})
                if response.status_code == 200:
                    data = response.json()
                    st.markdown(data["answer"])
                    
                    if data.get("sources"):
                        st.markdown("---")
                        for source in data["sources"]:
                            st.markdown(f'<span class="source-tag">📄 {source}</span>', unsafe_allow_html=True)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": data["answer"],
                        "sources": data.get("sources")
                    })
                else:
                    st.error("Backend returned an error.")
            except Exception as e:
                st.error(f"Connection failed: {e}")
