# In app/services/rag_service.py (Final Version)

import os
import json
from typing import List, Dict

# LangChain & Document Processing
from app.utils.document_loader import load_and_extract_text_from_url
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Google Gemini API
import google.generativeai as genai

# Configure the Gemini API client
try:
    genai.configure(api_key='AIzaSyC0j-L51sodUuYAuY7M-H7Ec0r7NqlRGjg')
except AttributeError:
    raise RuntimeError("GOOGLE_API_KEY is not set. Please check your .env file.")

# Initialize Gemini models
parser_model = genai.GenerativeModel('gemini-1.5-flash-latest')
adjudicator_model = genai.GenerativeModel('gemini-1.5-flash-latest')

# --- Helper Functions for RAG and LLM Interaction ---

def _create_vector_store(text: str):
    """Chunks text, creates embeddings, and stores them in a FAISS vector store."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vector_store

def _parse_query_to_json(question: str) -> Dict:
    """Uses Gemini to parse a natural language query into a structured JSON."""
    prompt = f"""
    Analyze the user's query and extract the key entities into a structured JSON object.
    The query is a set of facts about an insurance claim.
    The entities to extract are: 'age', 'gender', 'procedure', 'location', and 'policy_duration_months'.
    If a value is not present, use null.

    Query: "{question}"

    JSON Output:
    """
    try:
        response = parser_model.generate_content(prompt)
        # Clean up the response to get a valid JSON string
        json_string = response.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(json_string)
    except Exception as e:
        print(f"Error parsing query to JSON: {e}")
        # Return a fallback structure if parsing fails
        return {"procedure": question, "details": "Parsing failed"}

def _get_final_answer(structured_query: Dict, vector_store) -> Dict:
    """Retrieves context and uses Gemini to adjudicate the claim and generate a final response."""
    # --- Multi-Faceted Retrieval ---
    search_queries = [
        structured_query.get('procedure', ''),
        "waiting period for surgeries and diseases",
        "exclusions and conditions",
        "pre-existing disease policy"
    ]
    # Filter out empty search queries
    search_queries = [q for q in search_queries if q]
    
    retrieved_docs = []
    for query in search_queries:
        retrieved_docs.extend(vector_store.similarity_search(query, k=2))
    
    # Remove duplicate documents
    unique_docs = {doc.page_content for doc in retrieved_docs}
    context = "\n\n---\n\n".join(unique_docs)

    # --- Adjudicator Prompt ---
    prompt = f"""
    You are an expert AI insurance claims adjudicator. Your task is to make a decision based on the provided facts and policy clauses.

    **Facts of the Claim (JSON):**
    {json.dumps(structured_query, indent=2)}

    **Relevant Policy Clauses (Context):**
    ---
    {context}
    ---

    **Instructions:**
    1.  **Analyze the facts and context.**
    2.  **Perform a step-by-step evaluation:**
        a. First, determine if the medical procedure is generally covered under the policy.
        b. Second, identify all applicable waiting periods (e.g., initial 30-day, specific 24-month, pre-existing disease) and check if the user's policy duration meets these requirements. This is the most critical step.
        c. Third, check for any other conditions or exclusions that might apply.
    3.  **Make a final decision:** Based on your analysis, decide if the claim is "Approved" or "Rejected".
    4.  **Provide a clear justification:** Explain your reasoning step-by-step, citing the specific text from the policy clauses that support your decision.
    5.  **Format your output as a single, clean JSON object** with the keys "decision" and "justification". Do not include any other text or formatting.
    
    **JSON Output:**
    """
    
    try:
        response = adjudicator_model.generate_content(prompt)
        json_string = response.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(json_string)
    except Exception as e:
        print(f"Error in final answer generation: {e}")
        return {"decision": "Error", "justification": "Failed to generate a valid decision from the LLM."}


# --- Main Orchestrator Function ---

def process_query_and_documents(doc_url: str, questions: List[str]) -> List[str]:
    """
    The main service function to orchestrate the entire RAG pipeline.
    """
    # === Step 1: Load and Extract Text ===
    try:
        document_text = load_and_extract_text_from_url(doc_url)
    except (ConnectionError, ValueError) as e:
        return [str(e)] * len(questions)

    # === Step 2: Create Vector Store ===
    if not document_text:
        return ["The document is empty or text could not be extracted."] * len(questions)
    vector_store = _create_vector_store(document_text)

    # === Step 3: Process Each Question ===
    final_answers = []
    for question in questions:
        print(f"\nProcessing question: {question}")
        
        # Parse the query into structured data
        structured_query = _parse_query_to_json(question)
        print(f"Parsed Query: {structured_query}")
        
        # Get the final decision from the adjudicator
        result_json = _get_final_answer(structured_query, vector_store)
        print(f"Adjudicator Result: {result_json}")
        
        # Format the final answer string as required by the hackathon response
        # The sample response is a simple string, so we will combine decision and justification.
        final_answer_str = f"Decision: {result_json.get('decision', 'N/A')}. Justification: {result_json.get('justification', 'N/A')}"
        final_answers.append(final_answer_str)
        
    return final_answers