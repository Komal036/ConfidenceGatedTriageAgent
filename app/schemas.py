from typing import Optional
from pydantic import BaseModel, Field


class TicketCreate(BaseModel):
    subject: str = Field(..., min_length=3, max_length=200)
    description: str = Field(..., min_length=5)
    product: Optional[str] = None
    channel: Optional[str] = None


class TicketResponse(BaseModel):
    id: str
    subject: str
    description: str
    category: Optional[str]
    priority: Optional[str]
    status: str

    # Added : surfaces what the Retriever/Resolver decided, so the pipeline's reasoning is visible in the API response 
    # itself and not just in the AgentDecision audit rows.
    matched_issue: Optional[str] = None
    match_similarity: Optional[float] = None
    draft_resolution: Optional[str] = None
    tool_called: Optional[str] = None

    class Config:
        from_attributes = True