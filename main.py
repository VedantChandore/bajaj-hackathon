from document_ingestion import load_pdf_chunks, embed_and_store
from semantic_retriever import retrieve_top_k
from policy_parser import parse_query
from decision_engine import evaluate_claim
import json

# Load & embed document
chunks = load_pdf_chunks("data/BAJHLIP23020V012223.pdf")
chunk_embeddings = embed_and_store(chunks)

# Input query
user_query = "46-year-old male, arthroscopic knee surgery, 3-month-old insurance policy"

# Step 1: Parse
parsed_query = parse_query(user_query)
print("Parsed Query:", parsed_query)

# Step 2: Retrieve top policy clauses
top_chunks = retrieve_top_k(user_query, chunks, chunk_embeddings, k=5)

# Step 3: Evaluate decision
decision_json = evaluate_claim(parsed_query, "\n\n".join(top_chunks))
print("Decision:", decision_json)

# Save output
with open("output/result.json", "w") as f:
    f.write(decision_json)
