"""
Session storage management for the FAQ retriever application.
This module handles the storage and retrieval of session data.
"""

import time
from typing import Dict, Optional, List

from app.retriever.faq_retriever import FAQRetriever

# In-memory storage for active sessions
# Maps session_id to FAQRetriever instance
active_sessions: Dict[str, FAQRetriever] = {}

def get_session(session_id: str) -> Optional[FAQRetriever]:
    """
    Get a session by ID if it exists.
    
    Args:
        session_id: Session identifier
        
    Returns:
        FAQRetriever instance or None if not found
    """
    return active_sessions.get(session_id)

def store_session(session_id: str, retriever: FAQRetriever) -> None:
    """
    Store a session in the active sessions dictionary.
    
    Args:
        session_id: Session identifier
        retriever: FAQRetriever instance
    """
    active_sessions[session_id] = retriever

def remove_session(session_id: str) -> bool:
    """
    Remove a session from storage.
    
    Args:
        session_id: Session identifier
        
    Returns:
        True if session was removed, False if not found
    """
    if session_id in active_sessions:
        del active_sessions[session_id]
        return True
    return False

def list_all_sessions() -> List[str]:
    """
    Get a list of all active session IDs.
    
    Returns:
        List of session IDs
    """
    return list(active_sessions.keys())

def get_session_count() -> int:
    """
    Get the count of active sessions.
    
    Returns:
        Number of active sessions
    """
    return len(active_sessions)

def get_session_metadata(session_id: str) -> Optional[Dict]:
    """
    Get metadata about a session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Dictionary with session metadata or None if not found
    """
    retriever = get_session(session_id)
    if not retriever:
        return None
        
    return {
        "session_id": session_id,
        "interaction_count": len(retriever.context.full_history),
        "last_active": retriever.last_activity
    }

def get_all_sessions_metadata() -> List[Dict]:
    """
    Get metadata about all active sessions.
    
    Returns:
        List of dictionaries with session metadata
    """
    result = []
    for session_id, retriever in active_sessions.items():
        result.append({
            "session_id": session_id,
            "interaction_count": len(retriever.context.full_history),
            "last_active": retriever.last_activity
        })
    return result

def prune_inactive_sessions(max_inactive_time: int) -> List[str]:
    """
    Remove sessions that have been inactive for longer than max_inactive_time.
    
    Args:
        max_inactive_time: Maximum inactive time in seconds
        
    Returns:
        List of removed session IDs
    """
    current_time = time.time()
    sessions_to_remove = []
    
    for session_id, retriever in active_sessions.items():
        if current_time - retriever.last_activity > max_inactive_time:
            sessions_to_remove.append(session_id)
    
    for session_id in sessions_to_remove:
        remove_session(session_id)
    
    return sessions_to_remove