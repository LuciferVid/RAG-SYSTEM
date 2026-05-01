import re

class DocumentChunker:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text, metadata):
        """Chunks text into segments with overlap and attaches metadata."""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        if not text:
            return []

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Create chunk object
            chunk = {
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_id": f"{metadata['doc_id']}_{start}"
                }
            }
            chunks.append(chunk)
            
            # Move start pointer forward by chunk_size - overlap
            if end >= len(text):
                break
            start += (self.chunk_size - self.chunk_overlap)
            
        return chunks
