import faiss
import cohere
import pdfplumber
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os

SYSTEM_PROMPT = """
You are an expert document assistant for insurance policies.  
Your task is to answer natural-language questions by reading only the provided policy text.  

When you answer:
1. Pull facts directly from the context—do not hallucinate.  
2. Be concise and precise.  
3. If the question asks "yes/no," start with that, then give a brief explanation.  
4. If numeric limits or periods are involved (e.g. waiting periods, sub-limits), quote the exact figure.  
5. Always reference the clause or section title (e.g. "Def. 18: Grace Period") so we know where you found it.  

Output format (plain text):
Answer: <your concise answer>  
Reference: <clause or section heading>  
Explanation: <one-sentence rationale>

Now, here is the context and the user's question.
"""

# Configurations
COHERE_API_KEY = "Kwi33HNnmXRDCkO4j7FndNP3LATOoKX3yvoOdztK"
co = cohere.Client(COHERE_API_KEY)

# FastAPI app
app = FastAPI()

# Schema
class QueryInput(BaseModel):
    document_path: str   # Local filesystem path to the PDF
    questions: list[str]

# Helper: serialize table to text
def table_to_text(table: list[list[str]]) -> str:
    rows = ["\t".join([cell or "" for cell in row]) for row in table]
    return "\n".join(rows)

# Text chunking
def chunk_text(text: str, max_tokens: int = 500) -> list[str]:
    chunks, current = [], []
    for para in text.split("\n"):
        if len(" ".join(current)) + len(para) < max_tokens:
            current.append(para)
        else:
            chunks.append(" ".join(current))
            current = [para]
    if current:
        chunks.append(" ".join(current))
    return [chunk.strip() for chunk in chunks if chunk.strip()]

# PDF parsing with table extraction
def parse_pdf(path: str) -> list[str]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PDF file not found at {path}")

    full_text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            # Extract plain text
            text = page.extract_text() or ""
            # Extract tables, append as text blocks
            tables = page.extract_tables()
            for tbl in tables or []:
                text += "\n" + table_to_text(tbl) + "\n"
            full_text += text + "\n"
    return chunk_text(full_text)

# Build FAISS index
def build_faiss_index(chunks: list[str]):
    # Fix: Add input_type parameter for embeddings
    embeddings = co.embed(
        texts=chunks, 
        model="embed-english-v3.0", 
        input_type="search_document"  # Add this parameter
    ).embeddings
    
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))
    return index, embeddings

# Retrieve top-k matches
def retrieve_relevant_chunks(query: str, chunks: list[str], index, embeddings, k: int = 5) -> list[str]:
    # Fix: Add input_type parameter for query embedding
    query_embedding = co.embed(
        texts=[query], 
        model="embed-english-v3.0",
        input_type="search_query"  # Add this parameter
    ).embeddings[0]
    
    D, I = index.search(np.array([query_embedding]).astype('float32'), k)
    return [chunks[i] for i in I[0]]

# Generate answer using LLM
def generate_answer(query: str, context: str) -> str:
    prompt = SYSTEM_PROMPT + f"\nCONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nRESPONSE:"
    
    # ✅ Log the prompt and config
    print("======== Cohere DEBUG LOG ========")
    print(f"Prompt length: {len(prompt)} characters")
    print(f"Prompt preview:\n{prompt[:500]}...\n")  # Print first 500 chars
    print("Calling co.generate() with:")
    print("  model='command-r-plus'")
    print("  temperature=0.2")
    print("  max_tokens=200")

    try:
        # Fix: Use co.generate() instead of co.chat(), and fix response parsing
        response = co.generate(
            model="command-r-plus",  # Updated model name
            prompt=prompt,
            temperature=0.2,
            max_tokens=200
        )
        
        # Fix: Correct response parsing
        result = response.generations[0].text.strip()
        print(f"✅ Generated response: {result[:200]}...")
        
    except Exception as e:
        print("❌ Error during co.generate():", e)
        raise e

    print("======== END DEBUG ========")
    return result

# Process queries
def process_queries(questions: list[str], chunks: list[str]) -> dict:
    index, embeddings = build_faiss_index(chunks)
    answers = []
    for q in questions:
        try:
            top_chunks = retrieve_relevant_chunks(q, chunks, index, embeddings, k=5)
            context = "\n\n".join(top_chunks)
            answer = generate_answer(q, context)
            answers.append(answer)
        except Exception as e:
            print(f"Error processing question '{q}': {e}")
            answers.append(f"Error: Unable to process question - {str(e)}")
    return {"answers": answers}

# API endpoint
@app.post("/api/v1/hackrx/run")
def run_query(input: QueryInput):
    try:
        chunks = parse_pdf(input.document_path)   # ← uses local PDF path
        return process_queries(input.questions, chunks)
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Run server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)