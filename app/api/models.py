from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class QueryRequest(BaseModel):
    """Request model for submitting a query."""
    query: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    """Response model for a query answer."""
    session_id: str
    answer: str
    is_direct_match: bool
    similarity_score: float
    suggested_follow_ups: Optional[List[str]] = None

class SessionInfo(BaseModel):
    """Information about a session."""
    session_id: str
    interaction_count: int
    last_active: str

class SessionListResponse(BaseModel):
    """Response model for listing sessions."""
    sessions: List[SessionInfo]

class SessionHistoryItem(BaseModel):
    """A single interaction in a conversation history."""
    query: str
    response: str
    timestamp: float

class SessionHistoryResponse(BaseModel):
    """Response model for getting conversation history."""
    session_id: str
    interactions: List[SessionHistoryItem]

class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str