# integration_test.py
import unittest
import os
import shutil
import time
import uuid
from unittest.mock import MagicMock, patch
import sys

# Add the project root to sys.path if needed
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context_window import ContextWindow, cleanup_old_sessions
from faq_retriever import FAQRetriever, get_session_retriever, prune_inactive_sessions, active_sessions

class MockEmbeddings:
    """Mock for OpenAIEmbeddings."""
    def embed_query(self, query):
        # Return a simple mock embedding
        return [0.1] * 10

class MockLLM:
    """Mock for ChatOpenAI."""
    def invoke(self, prompt):
        # Return a simple mock response
        class MockResponse:
            @property
            def content(self):
                return f"This is a mock response to: {prompt[:50]}..."
        
        return MockResponse()

class MockFaiss:
    """Mock for FAISS index."""
    def __init__(self, ntotal=10):
        self.ntotal = ntotal
    
    def search(self, query_embedding, k):
        # Return mock distances and indices
        return [
            [0.2, 0.3, 0.4, 0.5, 0.6]  # Distances
        ], [
            [0, 1, 2, 3, 4]  # Indices
        ]

@patch('faiss.read_index')
@patch('json.load')
@patch('builtins.open', new_callable=unittest.mock.mock_open, read_data='{"questions": []}')
class IntegrationTest(unittest.TestCase):
    """Integration tests for the FAQRetriever with ContextWindow."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a test directory
        self.test_dir = f"test_integration_{uuid.uuid4()}"
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create data directory
        self.data_dir = os.path.join(self.test_dir, "data")
        self.context_dir = os.path.join(self.data_dir, "context")
        os.makedirs(self.context_dir, exist_ok=True)
        
        # Clear active sessions
        active_sessions.clear()
        
        # Set up mocks
        self.mock_embeddings = MockEmbeddings()
        self.mock_llm = MockLLM()
        self.mock_faiss = MockFaiss()
        
        # Store original import paths to restore later
        self.original_imports = {
            'DATA_DIR': getattr(sys.modules['faq_retriever'], 'DATA_DIR', None),
            'CONTEXT_DIR': getattr(sys.modules['faq_retriever'], 'CONTEXT_DIR', None),
        }
        
        # Set the data directories to our test directories
        setattr(sys.modules['faq_retriever'], 'DATA_DIR', self.data_dir)
        setattr(sys.modules['faq_retriever'], 'CONTEXT_DIR', self.context_dir)
    
    def tearDown(self):
        """Clean up after tests."""
        # Restore original import paths
        for name, value in self.original_imports.items():
            if value is not None:
                setattr(sys.modules['faq_retriever'], name, value)
        
        # Remove test directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_full_retriever_flow(self, mock_open, mock_json_load, mock_read_index):
        """Test the full flow of the FAQRetriever with context window."""
        # Set up mocks
        mock_read_index.return_value = self.mock_faiss
        mock_json_load.return_value = [
            {"type": "question", "question": "What is the meaning of life?"},
            {"type": "answer", "question": "What is the meaning of life?", "chunk": "The answer is 42."},
            {"type": "question", "question": "Who created Python?"},
            {"type": "answer", "question": "Who created Python?", "chunk": "Guido van Rossum."}
        ]
        
        # Create a context window directly to test it
        context = ContextWindow(max_size=3, session_id="test_session", storage_path=self.context_dir)
        
        # Patch the FAQRetriever to use our mocks
        with patch('langchain_openai.OpenAIEmbeddings', return_value=self.mock_embeddings), \
             patch('langchain_openai.ChatOpenAI', return_value=self.mock_llm):
            
            # Create a retriever
            retriever = FAQRetriever(
                context_window_size=3,
                session_id="test_session"
            )
            
            # Replace the context with our pre-created one for testing
            retriever.context = context
            
            # Simulate a conversation flow
            
            # First query
            response1 = retriever.get_answer("What is the meaning of life?")
            
            # Check that the response is as expected
            self.assertTrue(response1.startswith("This is a mock response"))
            
            # Check that the interaction was added to the context
            self.assertEqual(len(retriever.context.interactions), 1)
            self.assertEqual(retriever.context.interactions[0]["query"], "What is the meaning of life?")
            
            # Second query (follow-up)
            response2 = retriever.get_answer("Can you explain that further?")
            
            # Check that the response is as expected
            self.assertTrue(response2.startswith("This is a mock response"))
            
            # Check that both interactions are in the context
            self.assertEqual(len(retriever.context.interactions), 2)
            self.assertEqual(retriever.context.interactions[0]["query"], "What is the meaning of life?")
            self.assertEqual(retriever.context.interactions[1]["query"], "Can you explain that further?")
            
            # Third query (different topic)
            response3 = retriever.get_answer("Who created Python?")
            
            # Check that all three interactions are in the context
            self.assertEqual(len(retriever.context.interactions), 3)
            
            # Fourth query (should push out the first one due to max_size=3)
            response4 = retriever.get_answer("When was Python created?")
            
            # Check that we still have max_size interactions
            self.assertEqual(len(retriever.context.interactions), 3)
            
            # Check that the oldest interaction was pushed out
            self.assertEqual(retriever.context.interactions[0]["query"], "Can you explain that further?")
            self.assertEqual(retriever.context.interactions[1]["query"], "Who created Python?")
            self.assertEqual(retriever.context.interactions[2]["query"], "When was Python created?")
            
            # Test starting a new conversation
            retriever.new_conversation()
            
            # Check that the context is empty
            self.assertEqual(len(retriever.context.interactions), 0)
    
    def test_session_management_integration(self, mock_open, mock_json_load, mock_read_index):
        """Test session management with multiple sessions."""
        # Set up mocks
        mock_read_index.return_value = self.mock_faiss
        mock_json_load.return_value = [
            {"type": "question", "question": "What is the meaning of life?"},
            {"type": "answer", "question": "What is the meaning of life?", "chunk": "The answer is 42."}
        ]
        
        # Patch the FAQRetriever to use our mocks
        with patch('langchain_openai.OpenAIEmbeddings', return_value=self.mock_embeddings), \
             patch('langchain_openai.ChatOpenAI', return_value=self.mock_llm):
            
            # Create two sessions
            session1_id, session1_retriever = get_session_retriever()
            session2_id, session2_retriever = get_session_retriever()
            
            # Check that we have two different sessions
            self.assertNotEqual(session1_id, session2_id)
            
            # Add interactions to session 1
            session1_retriever.get_answer("What is the meaning of life?")
            session1_retriever.get_answer("Can you explain that further?")
            
            # Add interaction to session 2
            session2_retriever.get_answer("Who created Python?")
            
            # Check that each session has the correct interactions
            self.assertEqual(len(session1_retriever.context.interactions), 2)
            self.assertEqual(len(session2_retriever.context.interactions), 1)
            
            # Check the content of each session
            self.assertEqual(session1_retriever.context.interactions[0]["query"], "What is the meaning of life?")
            self.assertEqual(session2_retriever.context.interactions[0]["query"], "Who created Python?")
            
            # Switch to session 1 again
            switched_session_id, switched_retriever = get_session_retriever(session1_id)
            
            # Check that we got the same session
            self.assertEqual(switched_session_id, session1_id)
            
            # Add another interaction to session 1
            switched_retriever.get_answer("Another question")
            
            # Check that the interaction was added to session 1
            self.assertEqual(len(switched_retriever.context.interactions), 3)
            self.assertEqual(switched_retriever.context.interactions[2]["query"], "Another question")
            
            # And session 2 remains unchanged
            self.assertEqual(len(session2_retriever.context.interactions), 1)
    
    def test_persistence_integration(self, mock_open, mock_json_load, mock_read_index):
        """Test that sessions persist their context between runs."""
        # Set up mocks
        mock_read_index.return_value = self.mock_faiss
        mock_json_load.return_value = [{"type": "question", "question": "Test question"}]
        
        # Create a real context for testing persistence
        test_context = ContextWindow(
            max_size=5, 
            session_id="persistence_session", 
            storage_path=self.context_dir
        )
        
        # Add interactions
        test_context.add_interaction("Question 1", "Answer 1")
        test_context.add_interaction("Question 2", "Answer 2")
        
        # Force save to disk
        test_context._save_context()
        
        # Now create a new context window with the same session ID
        reloaded_context = ContextWindow(
            max_size=5, 
            session_id="persistence_session", 
            storage_path=self.context_dir
        )
        
        # Check that the interactions were loaded
        self.assertEqual(len(reloaded_context.interactions), 2)
        self.assertEqual(reloaded_context.interactions[0]["query"], "Question 1")
        self.assertEqual(reloaded_context.interactions[1]["query"], "Question 2")
        
        # Now test with the full retriever
        with patch('langchain_openai.OpenAIEmbeddings', return_value=self.mock_embeddings), \
             patch('langchain_openai.ChatOpenAI', return_value=self.mock_llm):
            
            # Create a retriever with the same session ID
            retriever = FAQRetriever(
                context_window_size=5,
                session_id="persistence_session"
            )
            
            # Check that the context was loaded
            self.assertEqual(len(retriever.context.interactions), 2)
            self.assertEqual(retriever.context.interactions[0]["query"], "Question 1")
            
            # Add a new interaction
            retriever.get_answer("Question 3")
            
            # Check that it was added to the existing context
            self.assertEqual(len(retriever.context.interactions), 3)
            self.assertEqual(retriever.context.interactions[2]["query"], "Question 3")



if __name__ == "__main__":
    unittest.main()
