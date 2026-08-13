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

    class Config:
        from_attributes = True
