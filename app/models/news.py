from pydantic import BaseModel
from typing import List

class NewsTransferRequest(BaseModel):
    title: str
    originalContent: str
    level: str

class DifficultWord(BaseModel):
    term: str
    description: str

class NewsTransferResult(BaseModel):
    level: str
    title: str
    transformedContent: str
    summarized: str
    difficultWords: List[DifficultWord]

class NewsTransferResponse(BaseModel):
    code: str
    message: str
    result: NewsTransferResult
    success: bool