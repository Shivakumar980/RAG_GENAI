import json
import os
import time
from typing import List, Dict, Any, Optional


class ContextWindow:
    """
    Context manager for chatbots that stores the complete conversation history
    while maintaining a sliding window of recent interactions for LLM context.
    """
    
    def __init__(self, max_size: int = 5, session_id: str = None, 
                 storage_path: Optional[str] = None):
        """
        Initialize the context window.
        
        Args:
            max_size: Maximum number of interactions to include in the LLM context
            session_id: Unique identifier for this conversation
            storage_path: Path to save context for persistence (None for no persistence)
        """
        self.max_size = max_size
        self.session_id = session_id or "default"
        self.storage_path = storage_path
        
        # Store all interactions
        self.full_history: List[Dict] = []
        
        # Store only the most recent interactions for LLM context
        self.context_window: List[Dict] = []
        
        # Load existing context if storage_path is provided
        if storage_path:
            self._load_context()
    
    def add_interaction(self, query: str, response: str) -> None:
        """
        Add a new interaction to both the full history and context window.
        Only the context window is limited by max_size.
        
        Args:
            query: The user's query
            response: The chatbot's response
        """
        # Create new interaction with timestamp
        new_interaction = {
            "query": query,
            "response": response,
            "timestamp": time.time()
        }
        
        # Add to full history (never truncated)
        self.full_history.append(new_interaction)
        
        # Add to context window (limited by max_size)
        self.context_window.append(new_interaction)
        
        # Remove oldest interaction from context window if we exceed max_size
        if len(self.context_window) > self.max_size:
            self.context_window.pop(0)
        
        # Save context if storage_path is provided
        if self.storage_path:
            self._save_context()
            self._update_session_metadata()
    
    def get_context_text(self) -> str:
        """
        Get the formatted context text for use in prompts.
        Uses only the recent interactions from the sliding window.
        
        Returns:
            Formatted string with recent interactions in chronological order
        """
        context_text = ""
        
        for interaction in self.context_window:
            context_text += f"User: {interaction['query']}\n"
            context_text += f"Assistant: {interaction['response']}\n\n"
            
        return context_text
    
    def get_full_history_text(self) -> str:
        """
        Get the complete conversation history as formatted text.
        
        Returns:
            Formatted string with all interactions in chronological order
        """
        history_text = ""
        
        for interaction in self.full_history:
            history_text += f"User: {interaction['query']}\n"
            history_text += f"Assistant: {interaction['response']}\n\n"
            
        return history_text
    
    def get_context_window(self) -> List[Dict]:
        """
        Get the raw context window data.
        
        Returns:
            List of recent interaction dictionaries
        """
        return self.context_window
    
    def get_full_history(self) -> List[Dict]:
        """
        Get the raw full history data.
        
        Returns:
            List of all interaction dictionaries
        """
        return self.full_history
    
    def clear(self) -> None:
        """Clear all interactions from both full history and context window."""
        self.full_history = []
        self.context_window = []
        if self.storage_path:
            self._save_context()
    
    def _get_storage_file(self, filename: str) -> str:
        """
        Get the full path to a storage file.
        
        Args:
            filename: Name of the file to get path for
            
        Returns:
            Full path to the storage file
        """
        if not self.storage_path:
            return None
            
        # Create a subdirectory based on the first 2 chars of session ID
        if len(self.session_id) >= 2:
            session_subdir = os.path.join(self.storage_path, self.session_id[:2])
        else:
            session_subdir = self.storage_path
            
        os.makedirs(session_subdir, exist_ok=True)
        return os.path.join(session_subdir, f"{self.session_id}_{filename}")
    
    def _save_context(self) -> None:
        """Save both full history and context window to disk."""
        if not self.storage_path:
            return
        
        # Save full history
        history_file = self._get_storage_file("history.json")
        with open(history_file, 'w') as f:
            json.dump(self.full_history, f)
        
        # Save context window
        context_file = self._get_storage_file("context.json")
        with open(context_file, 'w') as f:
            json.dump(self.context_window, f)
    
    def _load_context(self) -> None:
        """Load both full history and context window from disk."""
        # Load full history
        history_file = self._get_storage_file("history.json")
        if history_file and os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    self.full_history = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                self.full_history = []
        
        # Load context window
        context_file = self._get_storage_file("context.json")
        if context_file and os.path.exists(context_file):
            try:
                with open(context_file, 'r') as f:
                    self.context_window = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                # If context window file is missing or corrupted,
                # recreate it from the most recent interactions in full history
                self.context_window = self.full_history[-self.max_size:] if len(self.full_history) > 0 else []
        else:
            # If context window file doesn't exist,
            # initialize it from the most recent interactions in full history
            self.context_window = self.full_history[-self.max_size:] if len(self.full_history) > 0 else []
    
    def _update_session_metadata(self) -> None:
        """Update session metadata file with last activity timestamp."""
        if not self.storage_path:
            return
            
        metadata_file = os.path.join(self.storage_path, "session_metadata.json")
        
        # Load existing metadata
        metadata = {}
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            except:
                metadata = {}
        
        # Update this session's metadata
        metadata[self.session_id] = {
            "last_active": time.time(),
            "interaction_count": len(self.full_history),
            "context_size": len(self.context_window),
            "session_age": time.time() - self.full_history[0]["timestamp"] if self.full_history else 0
        }
        
        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)


def cleanup_old_sessions(storage_path: str, max_age_days: int = 30) -> int:
    """
    Remove context files for sessions older than max_age_days.
    
    Args:
        storage_path: Path to the context storage directory
        max_age_days: Maximum age in days for session files
        
    Returns:
        Number of session files removed
    """
    if not os.path.exists(storage_path):
        return 0
        
    current_time = time.time()
    max_age_seconds = max_age_days * 24 * 60 * 60
    removed_count = 0
    
    # Get session metadata
    metadata_file = os.path.join(storage_path, "session_metadata.json")
    metadata = {}
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        except:
            metadata = {}
    
    # Find and remove old session files
    for root, dirs, files in os.walk(storage_path):
        for file in files:
            if file.endswith("_history.json") or file.endswith("_context.json"):
                file_path = os.path.join(root, file)
                
                # Extract session ID from filename (remove _history.json or _context.json)
                session_id = file.split('_')[0]
                
                # Check if we have metadata for this session
                is_old = False
                if session_id in metadata:
                    last_active = metadata[session_id].get("last_active", 0)
                    is_old = (current_time - last_active) > max_age_seconds
                else:
                    # Fall back to file modification time
                    is_old = (current_time - os.path.getmtime(file_path)) > max_age_seconds
                
                if is_old:
                    try:
                        os.remove(file_path)
                        removed_count += 1
                    except:
                        pass
    
    # Update metadata file to remove entries for deleted sessions
    if metadata and removed_count > 0:
        # Verify each session still has a file
        updated_metadata = {}
        for session_id, session_data in metadata.items():
            # Check if any session file still exists
            if len(session_id) >= 2:
                session_subdir = os.path.join(storage_path, session_id[:2])
            else:
                session_subdir = storage_path
                
            history_path = os.path.join(session_subdir, f"{session_id}_history.json")
            context_path = os.path.join(session_subdir, f"{session_id}_context.json")
            
            if os.path.exists(history_path) or os.path.exists(context_path):
                updated_metadata[session_id] = session_data
        
        # Save updated metadata
        with open(metadata_file, 'w') as f:
            json.dump(updated_metadata, f)
    
    return removed_count