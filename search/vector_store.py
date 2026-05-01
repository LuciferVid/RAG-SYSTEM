import faiss
import numpy as np
import pickle
import os

class VectorStore:
    def __init__(self, 
                 index_path='data/vector_store/index.faiss', 
                 chunks_path='data/vector_store/chunks.pkl',
                 synced_files_path='data/vector_store/synced_files.pkl'):
        self.index_path = index_path
        self.chunks_path = chunks_path
        self.synced_files_path = synced_files_path
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        self.index = self._load_index()
        self.chunks = self._load_chunks()
        self.synced_files = self._load_synced_files()

    def _load_index(self):
        if os.path.exists(self.index_path):
            return faiss.read_index(self.index_path)
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

    def search(self, query_embedding, k=5):
        """Searches for top k relevant chunks."""
        if self.index.ntotal == 0:
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
