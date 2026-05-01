import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, storage_dir="data/vector_store"):
        self.storage_dir = storage_dir
        self.index_path = os.path.join(storage_dir, "index.faiss")
        self.chunks_path = os.path.join(storage_dir, "chunks.pkl")
        self.synced_files_path = os.path.join(storage_dir, "synced_files.pkl")
        self.dimension = 768  # Dimension for Gemini text-embedding-004
        
        # Ensure directory exists
        os.makedirs(self.storage_dir, exist_ok=True)
        
        self.index = self._load_index()
        self.chunks = self._load_chunks()
        self.synced_files = self._load_synced_files()

    def _load_index(self):
        if os.path.exists(self.index_path):
            try:
                return faiss.read_index(self.index_path)
            except Exception:
                pass
        return faiss.IndexFlatL2(self.dimension)

    def _load_chunks(self):
        if os.path.exists(self.chunks_path):
            with open(self.chunks_path, 'rb') as f:
                return pickle.load(f)
        return []

    def _load_synced_files(self):
        if os.path.exists(self.synced_files_path):
            with open(self.synced_files_path, 'rb') as f:
                return pickle.load(f)
        return {}

    def add_chunks(self, new_chunks):
        if not new_chunks:
            return
            
        import google.generativeai as genai
        
        texts = [chunk['text'] for chunk in new_chunks]
        
        # Batch requests to Gemini API (Limit to 100 per request usually, but let's do 50 for safety)
        batch_size = 50
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            try:
                response = genai.embed_content(
                    model="models/text-embedding-004",
                    content=batch_texts,
                    task_type="retrieval_document"
                )
                if 'embedding' in response:
                    all_embeddings.extend(response['embedding'])
            except Exception as e:
                print(f"Embedding error: {e}")
                # Fallback zero vectors if error occurs
                all_embeddings.extend([[0.0] * self.dimension] * len(batch_texts))
                
        if not all_embeddings:
            return
            
        embeddings = np.array(all_embeddings).astype('float32')
        
        self.index.add(embeddings)
        self.chunks.extend(new_chunks)
        
        self.save()

    def add_documents(self, embeddings, chunks, file_id=None, modified_time=None):
        """Adds embeddings and corresponding chunks to the store."""
        if len(embeddings) > 0:
            self.index.add(np.array(embeddings).astype('float32'))
            self.chunks.extend(chunks)
        
        if file_id and modified_time:
            self.synced_files[file_id] = modified_time
            
        self.save()

    def is_file_synced(self, file_id, modified_time):
        """Checks if a file is already synced and not modified."""
        return self.synced_files.get(file_id) == modified_time

    def search(self, query, k=5):
        """Searches for top k relevant chunks using a text query."""
        if self.index.ntotal == 0:
            return []
            
        import google.generativeai as genai
        try:
            response = genai.embed_content(
                model="models/text-embedding-004",
                content=query,
                task_type="retrieval_query"
            )
            query_embedding = response.get('embedding')
            if not query_embedding:
                return []
        except Exception as e:
            print(f"Query embedding error: {e}")
            return []
            
        distances, indices = self.index.search(np.array([query_embedding]).astype('float32'), k)
        
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.chunks):
                results.append(self.chunks[idx])
        return results

    def save(self):
        """Persists the index, chunks, and synced files to disk."""
        faiss.write_index(self.index, self.index_path)
        with open(self.chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        with open(self.synced_files_path, 'wb') as f:
            pickle.dump(self.synced_files, f)

    def clear(self):
        """Clears the store."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks = []
        self.synced_files = {}
        if os.path.exists(self.index_path): os.remove(self.index_path)
        if os.path.exists(self.chunks_path): os.remove(self.chunks_path)
        if os.path.exists(self.synced_files_path): os.remove(self.synced_files_path)
