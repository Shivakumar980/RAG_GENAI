# test_session_management.py
import os
import time
import shutil
import unittest
import json
from unittest.mock import MagicMock, patch
from faq_retriever import FAQRetriever, get_session_retriever, prune_inactive_sessions, active_sessions

class MockRetriever:
    """Mock version of FAQRetriever for testing without needing real components."""
    def __init__(self, session_id=None):
        self.session_id = session_id or "mock_session"
        self.last_activity = time.time()
        self.context = MagicMock()
        self.context.interactions = []
        self.context.add_interaction = MagicMock()
        self.context.get_context_text = MagicMock(return_value="Mock context text")
        self.context.clear = MagicMock()
        
    def get_answer(self, query):
        self.last_activity = time.time()
        self.context.add_interaction.assert_called_with(query, "Mock response")
        return "Mock response"
    
    def new_conversation(self):
        self.last_activity = time.time()
        self.context.clear.assert_called_once()
    
    def get_conversation_history(self):
        self.last_activity = time.time()
        return self.context.get_context_text()


class TestSessionManagement(unittest.TestCase):
    """Test cases for session management functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Clear active sessions
        active_sessions.clear()
        
        # Create a test directory
        self.test_dir = "test_session_storage"
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Set up mocks
        self.original_faq_retriever = FAQRetriever
        
        # Mock FAQRetriever with our simplified version
        self.retriever_mock = MagicMock()
        self.retriever_mock.return_value = MockRetriever()
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove test directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    @patch('faq_retriever.FAQRetriever')
    def test_get_session_retriever_new(self, mock_faq_retriever):
        """Test getting a new retriever with no specified session ID."""
        # Set up the mock
        mock_faq_retriever.return_value = MockRetriever()
        
        # Get a new retriever
        session_id, retriever = get_session_retriever()
        
        # Check that we got a non-empty session ID
        self.assertTrue(session_id and len(session_id) > 0)
        
        # Check that the retriever was created with this session ID
        mock_faq_retriever.assert_called_once()
        call_kwargs = mock_faq_retriever.call_args.kwargs
        self.assertEqual(call_kwargs['session_id'], session_id)
        
        # Check that the session was added to active_sessions
        self.assertIn(session_id, active_sessions)
    
    @patch('faq_retriever.FAQRetriever')
    def test_get_session_retriever_existing(self, mock_faq_retriever):
        """Test getting an existing retriever by session ID."""
        # Set up the mock
        mock_retriever = MockRetriever("test_session")
        mock_faq_retriever.return_value = mock_retriever
        
        # Add a mock retriever to active_sessions
        active_sessions["test_session"] = mock_retriever
        
        # Get the retriever by session ID
        session_id, retriever = get_session_retriever("test_session")
        
        # Check that we got the correct session ID
        self.assertEqual(session_id, "test_session")
        
        # Check that the retriever is the existing one
        self.assertEqual(retriever, mock_retriever)
        
        # Check that no new retriever was created
        mock_faq_retriever.assert_not_called()
    
    @patch('faq_retriever.FAQRetriever')
    def test_prune_inactive_sessions(self, mock_faq_retriever):
        """Test pruning inactive sessions."""
        # Create several mock retrievers with different activity times
        active_retriever = MockRetriever("active_session")
        inactive_retriever = MockRetriever("inactive_session")
        very_inactive_retriever = MockRetriever("very_inactive_session")
        
        # Set up activity times
        active_retriever.last_activity = time.time()  # Now
        inactive_retriever.last_activity = time.time() - 1800  # 30 minutes ago
        very_inactive_retriever.last_activity = time.time() - 7200  # 2 hours ago
        
        # Add to active_sessions
        active_sessions["active_session"] = active_retriever
        active_sessions["inactive_session"] = inactive_retriever
        active_sessions["very_inactive_session"] = very_inactive_retriever
        
        # Prune inactive sessions (older than 1 hour)
        removed = prune_inactive_sessions(max_inactive_time=3600)
        
        # Check that one session was removed
        self.assertEqual(removed, 1)
        
        # Check that the correct session was removed
        self.assertIn("active_session", active_sessions)
        self.assertIn("inactive_session", active_sessions)
        self.assertNotIn("very_inactive_session", active_sessions)
    
    @patch('faq_retriever.FAQRetriever')
    def test_multiple_sessions(self, mock_faq_retriever):
        """Test handling multiple concurrent sessions."""
        # Create mock retrievers for different sessions
        session1_retriever = MockRetriever("session1")
        session2_retriever = MockRetriever("session2")
        
        # Set up the mock to return different retrievers based on session ID
        def get_mock_retriever(session_id=None, **kwargs):
            if session_id == "session1":
                return session1_retriever
            elif session_id == "session2":
                return session2_retriever
            else:
                return MockRetriever(session_id)
        
        mock_faq_retriever.side_effect = get_mock_retriever
        
        # Get retrievers for two different sessions
        session1_id, retriever1 = get_session_retriever("session1")
        session2_id, retriever2 = get_session_retriever("session2")
        
        # Check that we got the correct retrievers
        self.assertEqual(retriever1, session1_retriever)
        self.assertEqual(retriever2, session2_retriever)
        
        # Check that both sessions are in active_sessions
        self.assertIn("session1", active_sessions)
        self.assertIn("session2", active_sessions)
        
        # Check that the retrievers are different
        self.assertNotEqual(retriever1, retriever2)
    
    @patch('faq_retriever.FAQRetriever')
    def test_session_isolation(self, mock_faq_retriever):
        """Test that sessions maintain isolated conversation contexts."""
        # Create mock retrievers
        session1_retriever = MockRetriever("session1")
        session2_retriever = MockRetriever("session2")
        
        # Add different interactions to each session
        session1_retriever.context.interactions = [
            {"query": "Question 1A", "response": "Answer 1A", "timestamp": time.time()},
            {"query": "Question 1B", "response": "Answer 1B", "timestamp": time.time()}
        ]
        
        session2_retriever.context.interactions = [
            {"query": "Question 2A", "response": "Answer 2A", "timestamp": time.time()}
        ]
        
        # Set up the mock
        def get_mock_retriever(session_id=None, **kwargs):
            if session_id == "session1":
                return session1_retriever
            elif session_id == "session2":
                return session2_retriever
            else:
                return MockRetriever(session_id)
        
        mock_faq_retriever.side_effect = get_mock_retriever
        
        # Add to active_sessions
        active_sessions["session1"] = session1_retriever
        active_sessions["session2"] = session2_retriever
        
        # Get the retrievers
        _, retriever1 = get_session_retriever("session1")
        _, retriever2 = get_session_retriever("session2")
        
        # Check that each retriever has its own context
        self.assertEqual(len(retriever1.context.interactions), 2)
        self.assertEqual(len(retriever2.context.interactions), 1)
        
        # Verify the content is different
        self.assertEqual(retriever1.context.interactions[0]["query"], "Question 1A")
        self.assertEqual(retriever2.context.interactions[0]["query"], "Question 2A")


if __name__ == "__main__":
    unittest.main()