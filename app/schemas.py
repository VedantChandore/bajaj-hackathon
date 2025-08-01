from pydantic import BaseModel
from typing import List

class HackathonRequest(BaseModel):
    documents: str  # This will be the URL to the PDF
    questions: List[str]

class HackathonResponse(BaseModel):
    answers: List[str]