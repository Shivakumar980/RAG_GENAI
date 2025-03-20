import json
import faiss
import numpy as np
import os
import time
from typing import Dict, List, Optional, Any, Tuple
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.context.context_window import ContextWindow
from app.retriever.embeddings import EmbeddingManager


class FAQRetriever:
    """
    Retrieval-augmented generation system for FAQ answering with context window support.
    """
    
    def __init__(self, 
                 model_name: str = settings.LLM_MODEL, 
                 temperature: float = settings.LLM_TEMPERATURE, 
                 top_k: int = settings.TOP_K, 
                 similarity_threshold: float = settings.SIMILARITY_THRESHOLD,
                 context_window_size: int = settings.CONTEXT_WINDOW_SIZE,
                 session_id: Optional[str] = None):
        """
        Initialize the FAQ retriever.
        
        Args:
            model_name: LLM model to use for response generation
            temperature: Temperature for response generation (higher = more creative)
            top_k: Number of related items to retrieve
            similarity_threshold: Threshold for direct matches
            context_window_size: Maximum conversation history to maintain
            session_id: Unique identifier for this conversation
        """
        # Initialize embedding manager
        self.embedding_manager = EmbeddingManager(model_name=settings.EMBEDDING_MODEL)
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            temperature=temperature,
            model=model_name
        )
        
        # Retrieval parameters
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.context_window_size = context_window_size
        
        # Session tracking
        self.session_id = session_id or str(time.time())
        self.last_activity = time.time()
        
        # Initialize context window
        self.context = ContextWindow(
            max_size=self.context_window_size,
            session_id=self.session_id,
            storage_path=settings.CONTEXT_DIR
        )
        
        # Load FAISS index and metadata
        self._load_faiss_index()
        self._load_metadata()
    
    def _load_faiss_index(self) -> None:
        """Load the FAISS index from disk."""
        try:
            self.index = faiss.read_index(settings.FAISS_INDEX_FILE)
        except Exception as e:
            raise ValueError(f"Failed to load FAISS index: {e}")
    
    def _load_metadata(self) -> None:
        """Load the metadata from disk."""
        try:
            with open(settings.METADATA_FILE, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load metadata: {e}")
    
    def retrieve(self, query: str, top_k: Optional[int] = None, 
                similarity_threshold: Optional[float] = None) -> Dict[str, List]:
        """
        Retrieve the most relevant FAQ chunks for the user query.
        
        Args:
            query: User's question
            top_k: Number of results to retrieve (optional)
            similarity_threshold: Threshold for direct matches (optional)
            
        Returns:
            Dictionary containing direct matches and related content
        """
        # Update last activity time
        self.last_activity = time.time()
        
        # Use provided parameters or defaults
        if top_k is None:
            top_k = self.top_k
        
        if similarity_threshold is None:
            similarity_threshold = self.similarity_threshold
            
        # Generate embedding for the query
        query_embedding = self.embedding_manager.get_embedding(query)
        query_embedding_array = np.array([query_embedding], dtype=np.float32)
        
        # Retrieve more results initially to allow for filtering
        expanded_k = min(top_k * 4, len(self.metadata))
        distances, indices = self.index.search(query_embedding_array, expanded_k)
        
        # Process the retrieved results
        direct_matches = []
        related_content = []
        seen_questions = set()
        
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # FAISS returns -1 if there aren't enough results
                metadata_entry = self.metadata[idx]
                similarity_score = 1.0 - distances[0][i]  # Convert distance to similarity (0-1)
                
                # Add similarity score to the metadata
                entry_with_score = metadata_entry.copy()
                entry_with_score["similarity_score"] = float(similarity_score)
                
                # Check if this is a direct question match
                if metadata_entry["type"] == "question" and similarity_score >= similarity_threshold:
                    direct_matches.append(entry_with_score)
                    seen_questions.add(metadata_entry["question"])
                
                # Store other relevant content
                elif len(related_content) < top_k:
                    # For answer chunks, check if we've already seen the question
                    if metadata_entry["type"] == "answer" and metadata_entry["question"] not in seen_questions:
                        related_content.append(entry_with_score)
                        seen_questions.add(metadata_entry["question"])
                    # For questions we haven't seen yet
                    elif metadata_entry["type"] == "question" and metadata_entry["question"] not in seen_questions:
                        related_content.append(entry_with_score)
                        seen_questions.add(metadata_entry["question"])
        
        return {
            "direct_matches": direct_matches,
            "related_content": related_content
        }
    
    def get_direct_answer(self, direct_matches: List[Dict], query: str) -> Optional[Dict]:
        """
        Get a concise answer for a direct question match, using conversation context.
        
        Args:
            direct_matches: List of direct matches from the retrieve method
            query: User's question
            
        Returns:
            Dictionary with answer and metadata, or None if no answer
        """
        if not direct_matches:
            return None
            
        # Sort by similarity score in descending order
        sorted_matches = sorted(direct_matches, key=lambda x: x["similarity_score"], reverse=True)
        top_match = sorted_matches[0]
        
        # Find the corresponding answer chunks
        answer_chunks = []
        for metadata_entry in self.metadata:
            if metadata_entry["type"] == "answer" and metadata_entry["question"] == top_match["question"]:
                answer_chunks.append(metadata_entry["chunk"])
        
        if not answer_chunks:
            return None
            
        # Combine answer chunks
        full_answer = " ".join(answer_chunks)
        
        # Get conversation context
        conversation_history = self.context.get_context_text()
        has_context = bool(conversation_history.strip())
            
        # For direct matches, generate a concise version using the LLM
        prompt = f"""You are a helpful assistant answering questions based on an FAQ knowledge base.

CONVERSATION HISTORY:
{conversation_history if has_context else "No previous conversation history."}

CURRENT QUESTION: "{query}"

FAQ MATCH: "{top_match["question"]}"

ANSWER FROM FAQ: "{full_answer}"

INSTRUCTIONS:
1. Provide a concise version of the answer (4-5 sentences maximum), focusing on the most important information.
2. Make sure to address the user's specific question without unnecessary details.
3. If this appears to be a follow-up question to something in the conversation history, acknowledge that connection.
4. If the user refers to something mentioned earlier (directly or indirectly), acknowledge and build upon that reference.
5. Use a conversational tone that maintains continuity with any previous exchanges.
6. If the user's current question contradicts or changes direction from earlier questions, acknowledge the change.
7. If you see relevant specific details from previous exchanges (like numbers, dates, names, or preferences mentioned), incorporate them in your response.

Remember to respond directly to the current question while demonstrating awareness of the conversation context.
"""
        response = self.llm.invoke(prompt)
        
        return {
            "question": top_match["question"],
            "answer": response.content,
            "full_answer": full_answer,
            "is_direct_match": True,
            "similarity_score": top_match["similarity_score"]
        }
    
    def format_context(self, content_list: List[Dict]) -> str:
        """
        Format retrieved results into a context string for the LLM.
        
        Args:
            content_list: List of content items from retrieve method
            
        Returns:
            Formatted context string
        """
        # Group answer chunks by question
        question_chunks = {}
        questions_only = []
        
        # First, collect all questions and their chunks
        for result in content_list:
            if result["type"] == "question":
                questions_only.append(result["question"])
            else:  # It's an answer chunk
                question = result["question"]
                if question not in question_chunks:
                    question_chunks[question] = []
                question_chunks[question].append(result["chunk"])
        
        # Now we need to get ALL chunks for each question from metadata
        # This ensures we have the complete answer, not just the chunks that matched
        for question in list(question_chunks.keys()):
            # Get all chunks for this question from metadata
            all_chunks = []
            for entry in self.metadata:
                if entry["type"] == "answer" and entry["question"] == question:
                    all_chunks.append(entry["chunk"])
            
            # Replace the partial chunks with all chunks if we found more
            if len(all_chunks) > len(question_chunks[question]):
                question_chunks[question] = all_chunks
        
        # Format the context as Q&A pairs
        context_parts = []
        
        # First add questions that were directly retrieved
        for question in questions_only:
            if question in question_chunks:
                answer_text = "".join(question_chunks[question])
                context_parts.append(f"Q: {question}")
                context_parts.append(f"A: {answer_text}")
        
        # Then add any remaining questions
        for question, chunks in question_chunks.items():
            if question not in questions_only:
                answer_text = "".join(chunks)
                context_parts.append(f"Q: {question}")
                context_parts.append(f"A: {answer_text}")
        
        return "\n\n".join(context_parts)
    
   

    def get_synthesized_answer(self, query: str, retrieval_results: Dict) -> Dict:
        """
        Generate a synthesized answer when no direct match is found.
        
        Args:
            query: User's question
            retrieval_results: Results from retrieve method
            
        Returns:
            Dictionary with answer and metadata
        """
        # Get related content
        related_content = retrieval_results["related_content"]
        
        # DEBUG: Print related content scores
        print(f"Query: '{query}'")
        print(f"Related content: {len(related_content)} items")
        if related_content:
            print(f"Similarity scores: {[f'{item['similarity_score']:.4f}' for item in related_content]}")
        else:
            print("No related content found")
        
        # Get conversation context
        conversation_history = self.context.get_context_text()
        has_context = bool(conversation_history.strip())
        
        # Check if we have any related content
        if not related_content:
            print("No related content - checking if answerable from conversation history")
            if has_context:
                # Let LLM try to answer from conversation history
                fallback_message = "I'm sorry, but that question isn't related to our knowledge base. Please ask a question about our services or policies, or contact our support team for further assistance."
                
                prompt = f"""You are a helpful assistant continuing a conversation with a user.

    CONVERSATION HISTORY:
    {conversation_history}

    CURRENT QUESTION: "{query}"

    INSTRUCTIONS:
    1. The user's question appears to be outside the scope of our knowledge base.
    2. Determine if you can provide a reasonable answer based ONLY on the previous conversation history.
    3. If you can answer the question using ONLY information from the conversation history, do so in 3-5 sentences.
    4. If you CANNOT answer the question using the conversation history (either because it introduces a new topic or requires information not discussed), respond with EXACTLY this message:
    "{fallback_message}"

    Do not add any additional explanations or apologies if you need to use the fallback message - use it exactly as provided.
    """
                
                # Generate the response
                response = self.llm.invoke(prompt)
                content = response.content.strip()
                
                # Check if the response is the fallback message
                is_fallback = fallback_message in content
                
                if not is_fallback:
                    print("Question answered from conversation history")
                    return {
                        "answer": content,
                        "is_direct_match": False,
                        "similarity_score": 0.3,
                        "source": "conversation_history"
                    }
                else:
                    print("Could not answer from conversation history - using fallback")
            
            # If no conversation history or couldn't answer from it
            return {
                "answer": "I'm sorry, but that question isn't related to our knowledge base. Please ask a question about our services or policies, or contact our support team for further assistance.",
                "is_direct_match": False,
                "similarity_score": 0.0
            }
        
        # Calculate average similarity score for the related content
        avg_score = sum(item["similarity_score"] for item in related_content) / len(related_content)
        print(f"Average similarity score: {avg_score:.4f}")
        
        # Check if average score is below threshold
        threshold = 0.5
        if avg_score < threshold:
            print(f"Average score ({avg_score:.4f}) below threshold ({threshold}) - checking if answerable from conversation history")
            
            if has_context:
                # Let LLM try to answer from conversation history
                fallback_message = "I'm sorry, but that question isn't related to our knowledge base. Please ask a question about our services or policies, or contact our support team for further assistance."
                
                prompt = f"""You are a helpful assistant continuing a conversation with a user.

    CONVERSATION HISTORY:
    {conversation_history}

    CURRENT QUESTION: "{query}"

    INSTRUCTIONS:
    1. The user's question appears to be outside the scope of our knowledge base.
    2. Determine if you can provide a reasonable answer based ONLY on the previous conversation history.
    3. If you can answer the question using ONLY information from the conversation history, do so in 3-5 sentences.
    4. If you CANNOT answer the question using the conversation history (either because it introduces a new topic or requires information not discussed), respond with EXACTLY this message:
    "{fallback_message}"

    Do not add any additional explanations or apologies if you need to use the fallback message - use it exactly as provided.
    """
                
                # Generate the response
                response = self.llm.invoke(prompt)
                content = response.content.strip()
                
                # Check if the response is the fallback message
                is_fallback = fallback_message in content
                
                if not is_fallback:
                    print("Question answered from conversation history")
                    return {
                        "answer": content,
                        "is_direct_match": False,
                        "similarity_score": 0.3,
                        "source": "conversation_history"
                    }
                else:
                    print("Could not answer from conversation history - using fallback")
            
            # If no conversation history or couldn't answer from it
            return {
                "answer": "I'm sorry, but that question isn't related to our knowledge base. Please ask a question about our services or policies, or contact our support team for further assistance.",
                "is_direct_match": False,
                "similarity_score": 0.0
            }
        
        # If we get here, the average score is above threshold
        print(f"Proceeding with LLM synthesis (avg score: {avg_score:.4f})")
        
        # Format context for the LLM
        faq_context = self.format_context(related_content)
        
        # Prepare the prompt for synthesizing a concise answer
        prompt = f"""You are a helpful assistant answering questions based on an FAQ knowledge base.

    CONVERSATION HISTORY:
    {conversation_history if has_context else "No previous conversation history."}

    CURRENT QUESTION: "{query}"

    RELEVANT FAQ ENTRIES:
    {faq_context}

    INSTRUCTIONS:
    1. The user's question doesn't exactly match any FAQ, but may be related to the RELEVANT FAQ entries above.
    2. Carefully analyze the current question in the context of the entire conversation history.
    3. If this is a follow-up question to something mentioned earlier, make that connection explicit in your response.
    4. If the user refers to something from earlier in the conversation (like "it", "that", "the question", or similar references), resolve what they're referring to before answering.
    5. Check if the current question is asking for clarification, more details, or an extension of a previous topic.
    6. Identify any user preferences, constraints, or specific details from previous exchanges that should influence your answer.
    7. Synthesize a concise answer (5-6 sentences maximum) that addresses the specific question while maintaining conversation flow.
    8. Begin your response with a sentence that directly addresses the current question while acknowledging relevant context.
    9. Focus only on the information provided in the FAQ entries above. Do not introduce information not contained in these entries.

    Remember: The most important factor in your response is to maintain natural conversation flow while accurately answering the current question using the FAQ information.
    """
        
        # Generate the response
        response = self.llm.invoke(prompt)
        
        return {
            "answer": response.content,
            "is_direct_match": False,
            "similarity_score": avg_score
        }
    def get_answer(self, query: str) -> Dict:
            """
            Generate an answer for the user query, considering conversation history.
            
            Args:
                query: User's question
                
            Returns:
                Dictionary with answer and metadata
            """
            # Update last activity time
            self.last_activity = time.time()
            
            # Retrieve relevant FAQ content
            retrieval_results = self.retrieve(query)
            
            # Get the maximum relevance score
            max_relevance = 0
            if retrieval_results["direct_matches"]:
                max_relevance = max(item["similarity_score"] for item in retrieval_results["direct_matches"])
            elif retrieval_results["related_content"]:
                max_relevance = max(item["similarity_score"] for item in retrieval_results["related_content"])
            
            # Check if we have a direct match
            direct_answer = None
            if retrieval_results["direct_matches"]:
                direct_answer = self.get_direct_answer(
                    retrieval_results["direct_matches"], 
                    query
                )
            
            result = {}
            if direct_answer:
                answer_text = direct_answer["answer"]
                result = {
                    "answer": answer_text,
                    "question": direct_answer["question"],
                    "is_direct_match": True,
                    "similarity_score": direct_answer["similarity_score"]
                }
            else:
                # For synthesized answers, include conversation context
                synthesized_answer = self.get_synthesized_answer(
                    query, 
                    retrieval_results
                )
                answer_text = synthesized_answer["answer"]
                result = {
                    "answer": answer_text,
                    "is_direct_match": False,
                    "similarity_score": synthesized_answer.get("similarity_score", 0)
                }
            
            # Add this exchange to the conversation history
            self.context.add_interaction(query, answer_text)
            
            return result
        
    def new_conversation(self) -> None:
        """Start a new conversation by clearing the context window."""
        self.context.clear()
        self.last_activity = time.time()
        
    def get_conversation_history(self) -> str:
        """
        Get the current conversation history as text.
        
        Returns:
            Formatted conversation history
        """
        self.last_activity = time.time()
        return self.context.get_context_text()
    
    def get_raw_history(self) -> List[Dict]:
        """
        Get the raw conversation history.
        
        Returns:
            List of interaction dictionaries
        """
        return self.context.get_raw_context()    