from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Optional


from app.api.models import (
    QueryRequest, 
    QueryResponse, 
    SessionInfo, 
    SessionListResponse,
    SessionHistoryItem,
    SessionHistoryResponse
)
from app.utils.session import (
    get_session_retriever, 
    list_sessions, 
    clear_session, 
    delete_session,
    prune_inactive_sessions
)
from app.context.context_window import cleanup_old_sessions
from app.core.config import settings

router = APIRouter(prefix="/api")

@router.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    Process a query and return the answer.
    
    Optionally include a session_id to continue an existing conversation.
    If no session_id is provided, a new session will be created.
    """
    # Run maintenance tasks in the background occasionally
    background_tasks.add_task(prune_inactive_sessions)
    
    # Get or create session
    session_id, retriever = get_session_retriever(request.session_id)
    
    # Process the query
    result = retriever.get_answer(request.query)
    
    # Return the response
    return {
        "session_id": session_id,
        "answer": result["answer"],
        "is_direct_match": result.get("is_direct_match", False),
        "similarity_score": result.get("similarity_score", 0.0),
        "suggested_follow_ups": None  # Could be implemented in future
    }

@router.get("/sessions", response_model=SessionListResponse)
async def get_sessions():
    """List all active sessions."""
    sessions = list_sessions()
    return {"sessions": sessions}

@router.post("/sessions", response_model=SessionInfo)
async def create_session():
    """Create a new session."""
    session_id, retriever = get_session_retriever()
    return {
        "session_id": session_id,
        "interaction_count": 0,
        "last_active": datetime.fromtimestamp(retriever.last_activity).isoformat()
    }

'''@router.delete("/sessions/{session_id}")
async def remove_session(session_id: str):
    """Delete a session."""
    if not delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "Session deleted"} '''

@router.delete("/sessions/{session_id}")
async def remove_session(session_id: str):
    """Delete a session and all its stored data."""
    if not delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "status": "Session deleted",
        "message": "Session and all associated data have been permanently removed"
    }

@router.post("/sessions/{session_id}/clear")
async def reset_session(session_id: str):
    """Clear a session's conversation history."""
    if not clear_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "Session cleared"}

'''@router.get("/sessions/{session_id}/history", response_model=SessionHistoryResponse)
async def get_session_history(session_id: str):
    """Get conversation history for a session."""
    # Get the session retriever
    try:
        session_id, retriever = get_session_retriever(session_id)
    except:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get the raw history
    raw_history = retriever.get_raw_history()
    
    # Format it for the response
    interactions = [
        SessionHistoryItem(
            query=item["query"],
            response=item["response"],
            timestamp=item["timestamp"]
        )
        for item in raw_history
    ]
    
    return {
        "session_id": session_id,
        "interactions": interactions
    }'''

@router.get("/sessions/{session_id}/history", response_model=SessionHistoryResponse)
async def get_session_history(session_id: str, full_history: bool = False):
    """
    Get conversation history for a session.
    
    Args:
        session_id: Session ID to get history for
        full_history: If True, return the complete history; if False, return only the context window
    """
    # Get the session retriever
    try:
        session_id, retriever = get_session_retriever(session_id)
    except:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get both history types to determine totals
    complete_history = retriever.context.get_full_history()
    context_window = retriever.context.get_context_window()
    
    # Get the appropriate history for the response
    raw_history = complete_history if full_history else context_window
    
    # Format it for the response
    interactions = [
        SessionHistoryItem(
            query=item["query"],
            response=item["response"],
            timestamp=item["timestamp"]
        )
        for item in raw_history
    ]
    
    return {
        "session_id": session_id,
        "interactions": interactions,
        "is_full_history": full_history,
        "total_interactions": len(complete_history)
    }    

@router.post("/maintenance")
async def run_maintenance(days: Optional[int] = None):
    """
    Run maintenance tasks to clean up old sessions.
    
    Args:
        days: Optional number of days to keep sessions (default from settings)
    """
    max_age = days or settings.SESSION_CLEANUP_DAYS
    
    # Clean up old session files
    removed_files = cleanup_old_sessions(settings.CONTEXT_DIR, max_age_days=max_age)
    
    # Clean up inactive sessions from memory
    removed_sessions = prune_inactive_sessions()
    
    return {
        "status": "Maintenance completed",
        "removed_files": removed_files,
        "removed_sessions": removed_sessions
    }