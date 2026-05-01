import io
from pypdf import PdfReader

class DocumentParser:
    @staticmethod
    def extract_text(content, mime_type):
        """Extracts text based on mime type."""
        try:
            if mime_type == 'application/pdf':
                return DocumentParser._parse_pdf(content)
            elif mime_type == 'text/plain' or mime_type == 'application/vnd.google-apps.document':
                return content.decode('utf-8')
            else:
                return ""
        except Exception as e:
            print(f"Error parsing document: {e}")
            return ""

    @staticmethod
    def _parse_pdf(content):
        reader = PdfReader(io.BytesIO(content))
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text
