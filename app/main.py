# In app/main.py (Updated version)

import os
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from .schemas import HackathonRequest, HackathonResponse
from .services.rag_service import process_query_and_documents # <-- IMPORT our new function

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="LLM Intelligent Query-Retrieval System",
    description="A system to answer questions from large documents using RAG and Gemini.",
    version="1.0.0"
)

# Authentication scheme
auth_scheme = HTTPBearer()
HACKATHON_TOKEN = os.getenv("HACKATHON_BEARER_TOKEN")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    """Dependency to verify the bearer token."""
    if not credentials or credentials.scheme != "Bearer" or credentials.credentials != HACKATHON_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials

@app.get("/", tags=["Health Check"])
def read_root():
    """Root endpoint to check if the API is running."""
    return {"status": "ok", "message": "Welcome to the Hackathon API!"}

@app.post("/api/v1/hackrx/run", response_model=HackathonResponse, tags=["Submission"])
async def run_submission(
    request: HackathonRequest,
    token: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """
    Main endpoint to process documents and questions.
    """
    try:
        # Call our main processing logic from rag_service
        answer_list = process_query_and_documents(request.documents, request.questions)
        
        return HackathonResponse(answers=answer_list)

    except Exception as e:
        # Log the error for debugging
        print(f"An error occurred in the main endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred: {str(e)}"
        )