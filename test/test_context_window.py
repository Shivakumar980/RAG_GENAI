# test_context_window.py
import os
import time
import json
import shutil
import unittest
from context_window import ContextWindow, cleanup_old_sessions

class TestContextWindow(unittest.TestCase):
    """Test cases for the ContextWindow class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a test directory for context storage
        self.test_dir = "test_context_storage"
        os.makedirs(self.test_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove the test directory and all its contents
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_add_interaction(self):
        """Test adding interactions to the context window."""
        # Create a context window with max_size=3
        context = ContextWindow(max_size=3, session_id="test_session", storage_path=self.test_dir)
        
        # Add interactions
        context.add_interaction("Question 1", "Answer 1")
        context.add_interaction("Question 2", "Answer 2")
        
        # Check that we have 2 interactions
        self.assertEqual(len(context.interactions), 2)
        
        # Check content of interactions
        self.assertEqual(context.interactions[0]["query"], "Question 1")
        self.assertEqual(context.interactions[0]["response"], "Answer 1")
        self.assertEqual(context.interactions[1]["query"], "Question 2")
        self.assertEqual(context.interactions[1]["response"], "Answer 2")
    
    def test_sliding_window(self):
        """Test that the sliding window removes oldest interactions when full."""
        # Create a context window with max_size=3
        context = ContextWindow(max_size=3, session_id="test_session", storage_path=self.test_dir)
        
        # Add 4 interactions (one more than max_size)
        context.add_interaction("Question 1", "Answer 1")
        context.add_interaction("Question 2", "Answer 2")
        context.add_interaction("Question 3", "Answer 3")
        context.add_interaction("Question 4", "Answer 4")
        
        # Check that we still have max_size interactions
        self.assertEqual(len(context.interactions), 3)
        
        # Check that the oldest interaction was removed
        self.assertEqual(context.interactions[0]["query"], "Question 2")
        self.assertEqual(context.interactions[1]["query"], "Question 3")
        self.assertEqual(context.interactions[2]["query"], "Question 4")
    
    def test_get_context_text(self):
        """Test getting formatted context text."""
        # Create a context window
        context = ContextWindow(max_size=5, session_id="test_session", storage_path=self.test_dir)
        
        # Add interactions
        context.add_interaction("Question 1", "Answer 1")
        context.add_interaction("Question 2", "Answer 2")
        
        # Get context text
        context_text = context.get_context_text()
        
        # Check that the text is formatted correctly
        expected_text = "User: Question 1\nAssistant: Answer 1\n\nUser: Question 2\nAssistant: Answer 2\n\n"
        self.assertEqual(context_text, expected_text)
    
    def test_clear(self):
        """Test clearing the context window."""
        # Create a context window
        context = ContextWindow(max_size=5, session_id="test_session", storage_path=self.test_dir)
        
        # Add interactions
        context.add_interaction("Question 1", "Answer 1")
        context.add_interaction("Question 2", "Answer 2")
        
        # Clear the context
        context.clear()
        
        # Check that we have 0 interactions
        self.assertEqual(len(context.interactions), 0)
    
    def test_persistence(self):
        """Test that context is persisted to disk and can be reloaded."""
        # Create a context window with persistence
        context1 = ContextWindow(max_size=5, session_id="test_session", storage_path=self.test_dir)
        
        # Add interactions
        context1.add_interaction("Question 1", "Answer 1")
        context1.add_interaction("Question 2", "Answer 2")
        
        # Create a new context window with the same session_id and storage_path
        context2 = ContextWindow(max_size=5, session_id="test_session", storage_path=self.test_dir)
        
        # Check that the interactions were loaded
        self.assertEqual(len(context2.interactions), 2)
        self.assertEqual(context2.interactions[0]["query"], "Question 1")
        self.assertEqual(context2.interactions[1]["query"], "Question 2")
    
    def test_session_specific_storage(self):
        """Test that different sessions have their own storage."""
        # Create two context windows with different session IDs
        context1 = ContextWindow(max_size=5, session_id="session1", storage_path=self.test_dir)
        context2 = ContextWindow(max_size=5, session_id="session2", storage_path=self.test_dir)
        
        # Add different interactions to each
        context1.add_interaction("Question 1A", "Answer 1A")
        context2.add_interaction("Question 2A", "Answer 2A")
        
        # Create new instances with the same session IDs
        context1b = ContextWindow(max_size=5, session_id="session1", storage_path=self.test_dir)
        context2b = ContextWindow(max_size=5, session_id="session2", storage_path=self.test_dir)
        
        # Check that each loaded its own interactions
        self.assertEqual(context1b.interactions[0]["query"], "Question 1A")
        self.assertEqual(context2b.interactions[0]["query"], "Question 2A")
    
    def test_session_metadata(self):
        """Test that session metadata is updated."""
        # Create a context window
        context = ContextWindow(max_size=5, session_id="test_session", storage_path=self.test_dir)
        
        # Add interactions
        context.add_interaction("Question 1", "Answer 1")
        context.add_interaction("Question 2", "Answer 2")
        
        # Check that metadata file exists
        metadata_file = os.path.join(self.test_dir, "session_metadata.json")
        self.assertTrue(os.path.exists(metadata_file))
        
        # Check metadata content
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Verify session is in metadata
        self.assertIn("test_session", metadata)
        
        # Verify metadata fields
        self.assertIn("last_active", metadata["test_session"])
        self.assertIn("interaction_count", metadata["test_session"])
        self.assertEqual(metadata["test_session"]["interaction_count"], 2)
    
    def test_cleanup_old_sessions(self):
        """Test cleaning up old session files."""
        # Create some context windows
        context1 = ContextWindow(max_size=5, session_id="active_session", storage_path=self.test_dir)
        context2 = ContextWindow(max_size=5, session_id="old_session", storage_path=self.test_dir)
        
        # Add interactions
        context1.add_interaction("Question 1", "Answer 1")
        context2.add_interaction("Question 2", "Answer 2")
        
        # Manually modify the last_active time in metadata to make second session old
        metadata_file = os.path.join(self.test_dir, "session_metadata.json")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Set the second session to be a week old
        week_ago = time.time() - (7 * 24 * 60 * 60)
        metadata["old_session"]["last_active"] = week_ago
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        # Run cleanup with max_age_days=1 (anything older than a day should be removed)
        removed = cleanup_old_sessions(self.test_dir, max_age_days=1)
        
        # Verify that one session was removed
        self.assertEqual(removed, 1)
        
        # Verify that the old session file is gone
        old_session_file = os.path.join(self.test_dir, "ol", "old_session_context.json")
        active_session_file = os.path.join(self.test_dir, "ac", "active_session_context.json")
        
        self.assertFalse(os.path.exists(old_session_file))
        self.assertTrue(os.path.exists(active_session_file))


if __name__ == "__main__":
    unittest.main()