# GDrive Intelligence RAG

A powerful Retrieval-Augmented Generation (RAG) system that transforms your Google Drive into an interactive knowledge base. Built for the Highwatch AI Platform Engineer trial.

## Features
- **Seamless GDrive Integration**: Supports OAuth 2.0 and Service Accounts.
- **Intelligent Processing**: Automatic text extraction from PDFs, Google Docs, and TXT files.
- **Semantic Search**: Powered by `SentenceTransformers` (`all-MiniLM-L6-v2`) and `FAISS`.
- **Grounded Q&A**: Uses Gemini 1.5 Flash for accurate, source-backed answers.
- **Exceptional Features**:
    - **Incremental Sync**: Only processes new or modified files.
    - **Batch Processing**: Efficient embedding generation.
    - **Async Pipeline**: Background synchronization tasks.

## Setup

### 1. Prerequisites
- Python 3.9+
- Google Cloud Project with Drive API enabled.
- Gemini API Key.

### 2. Credentials
Place your credentials in `data/credentials/`:
- `service_account.json` (for Service Account access)
- OR `credentials.json` (for OAuth access)

### 3. Installation
```bash
pip install -r requirements.txt
```

### 4. Configuration
Create a `.env` file:
```env
GEMINI_API_KEY=your_api_key_here
```

### 5. Running
```bash
python -m api.main
```

## API Endpoints
- `POST /sync-drive`: Triggers incremental sync.
- `POST /ask`: Query the knowledge base.
    ```json
    { "query": "What is our compliance policy?" }
    ```
- `GET /status`: Check indexing status.

## Architecture
- `connectors/`: Google Drive API interface.
- `processing/`: Text extraction and chunking logic.
- `embedding/`: Vector representation layer.
- `search/`: Vector database (FAISS) management.
- `api/`: FastAPI endpoints and LLM integration.
