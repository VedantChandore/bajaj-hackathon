import os
from dotenv import load_dotenv
load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
COHERE_MODEL = "command-r"  # Or use "command-r-plus" if available

EMBED_MODEL = "embed-english-v3.0"
PDF_PATH = "data/BAJHLIP23020V012223.pdf"
