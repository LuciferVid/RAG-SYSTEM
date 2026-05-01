import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

class LLMService:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("Warning: GEMINI_API_KEY not found in environment variables.")
        else:
            genai.configure(api_key=api_key)
        
        self.model = genai.GenerativeModel('models/gemini-flash-latest')

    def generate_answer(self, query, context_chunks, system_metadata=""):
        """Generates an answer based on the provided context."""
        if not context_chunks and not system_metadata:
            return "I don't have enough information in your Google Drive to answer this question."

        context_text = "\n\n".join([
            f"Source: {c['metadata']['file_name']}\nContent: {c['text']}" 
            for c in context_chunks
        ])
        
        prompt = f"""
        You are a helpful AI assistant connected to the user's Google Drive. 
        Answer the following question based ONLY on the provided context or system metadata.
        If the answer is not in the context, say that you don't have enough information in your Google Drive.
        
        System Metadata:
        {system_metadata}
        
        Document Context:
        {context_text}
        
        Question: {query}
        
        Instructions:
        - Be concise and accurate.
        - Cite your sources by mentioning the file names if you use Document Context.
        - Return the answer in clear paragraphs.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating answer: {str(e)}"
