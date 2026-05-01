from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import asyncio
from dotenv import load_dotenv

# Import our custom modules
from connectors.gdrive import GDriveConnector
from processing.parser import DocumentParser
from processing.chunker import DocumentChunker
from embedding.model import EmbeddingModel
from search.vector_store import VectorStore
from api.llm_service import LLMService

load_dotenv()

app = FastAPI(title="GDrive Intelligence RAG")

# Global instances
embedding_model = EmbeddingModel()
vector_store = VectorStore()
chunker = DocumentChunker()
llm_service = LLMService()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

@app.get("/status")
async def status():
    return {
        "indexed_chunks": len(vector_store.chunks),
        "synced_files_count": len(vector_store.synced_files),
        "ready": True
    }

@app.post("/sync-drive")
async def sync_drive(background_tasks: BackgroundTasks):
    """Triggers incremental synchronization from Google Drive."""
    try:
        connector = GDriveConnector()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    background_tasks.add_task(run_sync, connector)
    return {"message": "Synchronization started in background."}

def run_sync(connector: GDriveConnector):
    """Async synchronization task."""
    print("Fetching files from Google Drive...")
    files = connector.list_files()
    
    for file in files:
        file_id = file['id']
        modified_time = file['modifiedTime']
        
        # Incremental Sync Check
        if vector_store.is_file_synced(file_id, modified_time):
            print(f"Skipping {file['name']} (already synced).")
            continue
            
        print(f"Processing {file['name']}...")
        try:
            content = connector.download_file(file_id, file['mimeType'])
            text = DocumentParser.extract_text(content, file['mimeType'])
            
            if not text:
                continue
                
            metadata = {
                "doc_id": file_id,
                "file_name": file['name'],
                "source": "gdrive"
            }
            
            chunks = chunker.chunk_text(text, metadata)
            if not chunks:
                continue
                
            # Batch process embeddings for the file
            chunk_texts = [c['text'] for c in chunks]
            embeddings = embedding_model.generate_embeddings(chunk_texts)
            
            # Add to vector store
            vector_store.add_documents(embeddings, chunks, file_id, modified_time)
            print(f"Synced {file['name']} ({len(chunks)} chunks).")
            
        except Exception as e:
            print(f"Error processing {file['name']}: {e}")

@app.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    """Retrieves context and generates an answer."""
    if not vector_store.chunks:
        raise HTTPException(status_code=400, detail="No documents indexed. Please run /sync-drive first.")
    
    # 1. Embed query
    query_embedding = embedding_model.generate_embeddings(request.query)[0]
    
    # 2. Search relevant chunks
    relevant_chunks = vector_store.search(query_embedding, k=5)
    
    if not relevant_chunks:
        return QueryResponse(answer="I couldn't find any relevant information in your documents.", sources=[])
    
    # 3. Generate answer
    answer = llm_service.generate_answer(request.query, relevant_chunks)
    
    # 4. Extract unique sources
    sources = list(set([c['metadata']['file_name'] for c in relevant_chunks]))
    
    return QueryResponse(answer=answer, sources=sources)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
