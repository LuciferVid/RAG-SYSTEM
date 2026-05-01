import os
from connectors.gdrive import GDriveConnector
from api.llm_service import LLMService
from dotenv import load_dotenv

load_dotenv()

def test_all():
    print("--- Testing GDrive Connection ---")
    try:
        connector = GDriveConnector()
        files = connector.list_files()
        print(f"Success! Found {len(files)} files.")
        for f in files[:5]:
            print(f" - {f['name']} ({f['mimeType']})")
        if not files:
            print("Warning: No files found. Make sure you've shared files with: aiml-rag@aiml-495007.iam.gserviceaccount.com")
    except Exception as e:
        print(f"GDrive Error: {e}")

    print("\n--- Testing Gemini Connection ---")
    try:
        llm = LLMService()
        response = llm.generate_answer("Say 'Hello GDrive RAG is ready!'", [])
        print(f"Gemini Response: {response}")
    except Exception as e:
        print(f"Gemini Error: {e}")

if __name__ == "__main__":
    test_all()
