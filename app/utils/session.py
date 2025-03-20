"""
Session management utilities for the FAQ retriever application.
This module provides high-level session management functions.
"""
import os
import json
import uuid
import time
from typing import Dict, List, Tuple, Any
from datetime import datetime

from app.retriever.faq_retriever import FAQRetriever
from app.core.config import settings
from app.utils import session_storage

def get_session_retriever(session_id=None) -> Tuple[str, FAQRetriever]:
    """
    Get or create a FAQRetriever for the specified session ID.
    
    Args:
        session_id: Unique session identifier, or None to create a new session
        
    Returns:
        Tuple of (session_id, FAQRetriever instance)
    """
    # Generate a new session ID if none provided
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # Check if session exists
    retriever = session_storage.get_session(session_id)
    
    # Create a new retriever for this session if it doesn't exist
    if not retriever:
        retriever = FAQRetriever(session_id=session_id)
        session_storage.store_session(session_id, retriever)
    
    return session_id, retriever

def list_sessions() -> List[Dict[str, Any]]:
    """List all active sessions with metadata."""
    result = []
    # Print debug info
    print(f"Active sessions: {list(session_storage.active_sessions.keys())}")
    
    # Directly iterate over the actual storage
    for session_id, retriever in session_storage.active_sessions.items():
        result.append({
            "session_id": session_id,
            "interaction_count": len(retriever.context.full_history),
            "last_active": datetime.fromtimestamp(retriever.last_activity).isoformat()
        })
    return result

def clear_session(session_id: str) -> bool:
    """
    Clear a session's context.
    
    Args:
        session_id: Session ID to clear
        
    Returns:
        True if successful, False if session not found
    """
    retriever = session_storage.get_session(session_id)
    if not retriever:
        return False
    
    retriever.new_conversation()
    return True

'''def delete_session(session_id: str) -> bool:
    """
    Delete a session entirely.
    
    Args:
        session_id: Session ID to delete
        
    Returns:
        True if successful, False if session not found
    """
    return session_storage.remove_session(session_id)'''


def delete_session(session_id: str) -> bool:
    """
    Delete a session entirely including all stored files.
    
    Args:
        session_id: Session ID to delete
        
    Returns:
        True if successful, False if session not found
    """
    # Get the retriever before removing it from memory
    retriever = session_storage.get_session(session_id)
    if not retriever:
        return False
    
    # Remove from in-memory storage
    success = session_storage.remove_session(session_id)
    
    # Also delete files from disk
    if success and retriever.context.storage_path:
        try:
            # Get paths to history and context files
            history_file = retriever.context._get_storage_file("history.json")
            context_file = retriever.context._get_storage_file("context.json")
            
            # Delete files if they exist
            if os.path.exists(history_file):
                os.remove(history_file)
            if os.path.exists(context_file):
                os.remove(context_file)
                
            # Update metadata file to remove this session
            metadata_file = os.path.join(retriever.context.storage_path, "session_metadata.json")
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Remove session from metadata if it exists
                    if session_id in metadata:
                        del metadata[session_id]
                        
                    # Save updated metadata
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f)
                except:
                    pass  # If metadata update fails, continue anyway
        except:
            pass  # If file deletion fails, the session was still removed from memory
    
    return success

def prune_inactive_sessions(max_inactive_time=settings.SESSION_TIMEOUT) -> int:
    """
    Remove retrievers for sessions that have been inactive.
    
    Args:
        max_inactive_time: Maximum inactive time in seconds before removing a session
        
    Returns:
        Number of sessions removed
    """
    removed_sessions = session_storage.prune_inactive_sessions(max_inactive_time)
    return len(removed_sessions)