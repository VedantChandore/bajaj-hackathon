# In app/utils/document_loader.py

import requests
import io
from pypdf import PdfReader

def load_and_extract_text_from_url(url: str) -> str:
    """
    Downloads a PDF from a URL, extracts its text content, and returns it as a single string.
    """
    print(f"Fetching document from URL: {url}")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)

        # Use an in-memory stream to handle the PDF content
        pdf_file = io.BytesIO(response.content)
        
        # Initialize the PDF reader
        pdf_reader = PdfReader(pdf_file)
        
        # Extract text from all pages
        full_text = ""
        for page in pdf_reader.pages:
            full_text += page.extract_text() + "\n"
            
        print(f"Successfully extracted {len(full_text)} characters from the document.")
        return full_text

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
        raise ConnectionError(f"Failed to download document from URL: {url}") from e
    except Exception as e:
        print(f"An error occurred during PDF processing: {e}")
        raise ValueError("Failed to process the PDF file. It might be corrupted or not a valid PDF.") from e