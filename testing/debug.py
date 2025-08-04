import cohere

co = cohere.Client("your-api-key-here")  # Replace with your key

SYSTEM_PROMPT = "You are an expert in insurance documents."

prompt = SYSTEM_PROMPT + "\nCONTEXT:\nThis policy covers cataract surgery.\n\nQUESTION:\nDoes this policy cover cataract surgery?\n\nRESPONSE:"

response = co.generate(
    model="command-r-plus",
    prompt=prompt,
    max_tokens=100,
    temperature=0.2,
    #error not solved
    input_type="prompt" # type: ignore 
)

print(response.generations[0].text)
