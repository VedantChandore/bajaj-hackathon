from config import COHERE_API_KEY, COHERE_MODEL
import cohere

client = cohere.Client(COHERE_API_KEY)

def parse_query(raw_query):
    prompt = f"""Extract the following fields from the user query:
- age
- gender
- procedure
- location
- policy_duration_months

Query: "{raw_query}"
Return in JSON format."""
    
    response = client.chat(
    model=COHERE_MODEL,
    message=prompt,
    temperature=0.3
)
    return response.text.strip()

   
