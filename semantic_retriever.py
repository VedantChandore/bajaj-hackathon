from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from config import COHERE_API_KEY, EMBED_MODEL
import cohere

client = cohere.Client(COHERE_API_KEY)

def embed_chunks(chunks):
    return client.embed(texts=chunks, model=EMBED_MODEL, input_type="search_document").embeddings

def retrieve_top_k(query, chunks, chunk_embeddings, k=5):
    query_embedding = client.embed(texts=[query], model=EMBED_MODEL, input_type="search_query").embeddings[0]
    sims = cosine_similarity([query_embedding], chunk_embeddings)[0]
    top_indices = np.argsort(sims)[::-1][:k]
    return [chunks[i] for i in top_indices]
