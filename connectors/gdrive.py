import os.path
import io
import json
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

class GDriveConnector:
    def __init__(self, credentials_path="data/credentials/service_account.json"):
        self.creds = self._authenticate(credentials_path)
        self.service = build('drive', 'v3', credentials=self.creds)

    def _authenticate(self, credentials_path):
        import streamlit as st
        
        # 1. Try Streamlit Secrets (for Cloud Hosting)
        if "gdrive_service_account" in st.secrets:
            creds_info = dict(st.secrets["gdrive_service_account"])
            return service_account.Credentials.from_service_account_info(creds_info, scopes=SCOPES)

        # 2. Try Service Account File
        if os.path.exists(credentials_path):
            return service_account.Credentials.from_service_account_file(credentials_path, scopes=SCOPES)

        # 2. Fallback: Local OAuth (token.json)
        token_path = "data/credentials/token.json"
        client_secrets_path = "data/credentials/credentials.json"
        
        creds = None
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            elif os.path.exists(client_secrets_path):
                flow = InstalledAppFlow.from_client_secrets_file(client_secrets_path, SCOPES)
                creds = flow.run_local_server(port=0)
                # Save the credentials for the next run
                with open(token_path, 'w') as token:
                    token.write(creds.to_json())
            else:
                raise Exception("No valid credentials found. Please provide service_account.json or credentials.json")
        
        return creds

    def list_files(self, query=None):
        """Lists files from Google Drive with optional query."""
        if not query:
            # Default: PDFs, Google Docs, and Text files
            query = "mimeType = 'application/pdf' or mimeType = 'application/vnd.google-apps.document' or mimeType = 'text/plain'"
        
        results = self.service.files().list(
            q=query, 
            pageSize=100, 
            fields="nextPageToken, files(id, name, mimeType, modifiedTime)"
        ).execute()
        return results.get('files', [])

    def download_file(self, file_id, mime_type):
        """Downloads file content. Converts Google Docs to text/plain."""
        if mime_type == 'application/vnd.google-apps.document':
            request = self.service.files().export_media(fileId=file_id, mimeType='text/plain')
        else:
            request = self.service.files().get_media(fileId=file_id)
        
        file_content = io.BytesIO()
        downloader = MediaIoBaseDownload(file_content, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        return file_content.getvalue()
