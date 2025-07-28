from config import COHERE_API_KEY, COHERE_MODEL
import cohere

client = cohere.Client(COHERE_API_KEY)

def evaluate_claim(parsed_query_json, retrieved_chunks):
    prompt = f"""You are an insurance expert. Based on the structured query and the relevant policy clauses, decide:

- decision: approved/rejected
- amount: max sum insured if available
- justification: refer to clause texts

Query:
{parsed_query_json}

Relevant Clauses:
{retrieved_chunks}

Return in JSON format with keys: decision, amount, justification.
"""
    response = client.chat(
    model=COHERE_MODEL,
    message=prompt,
    temperature=0.2
)
    return response.text.strip()

