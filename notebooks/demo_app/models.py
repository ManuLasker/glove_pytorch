from typing import Optional, List
from pydantic import BaseModel

class SearchResponseModel(BaseModel):
    results: List[str] = []

class PingResponseModel(BaseModel):
    status: str