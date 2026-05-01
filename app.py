import streamlit as st
import requests
import time
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8002")

st.set_page_config(
    page_title="GDrive Intelligence RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a "Premium" look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
    }
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
        with st.spinner("Syncing... (Check terminal for progress)"):
            try:
                response = requests.post(f"{API_URL}/sync-drive")
                if response.status_code == 200:
                    st.success("Sync started in background!")
                else:
                    st.error("Failed to start sync.")
            except Exception as e:
                st.error(f"Error: {e}")
    
    st.markdown("---")
    st.subheader("Indexing Status")
    try:
        status = requests.get(f"{API_URL}/status").json()
        st.metric("Indexed Chunks", status.get("indexed_chunks", 0))
        st.metric("Synced Files", status.get("synced_files_count", 0))
    except:
        st.warning("Could not reach API server.")

# Main Interface
st.title("Drive Intelligence Assistant")
st.info("Ask anything about your shared Google Drive documents.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            cols = st.columns(len(message["sources"]))
            for idx, source in enumerate(message["sources"]):
                st.markdown(f'<span class="source-tag">📄 {source}</span>', unsafe_allow_html=True)

# React to user input
if prompt := st.chat_input("What would you like to know?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"query": prompt}
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data["sources"]
                    
                    st.markdown(answer)
                    if sources:
                        st.markdown("---")
                        st.markdown("**Sources:**")
                        for source in sources:
                            st.markdown(f'<span class="source-tag">📄 {source}</span>', unsafe_allow_html=True)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })
                else:
                    st.error("API returned an error.")
            except Exception as e:
                st.error(f"Connection Error: {e}")
