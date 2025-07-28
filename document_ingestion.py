import fitz  # PyMuPDF
from semantic_retriever import embed_chunks

def load_pdf_chunks(path, chunk_size=300):
    doc = fitz.open(path)
    text = " ".join([page.get_text() for page in doc])
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def embed_and_store(chunks):
    return embed_chunks(chunks)
