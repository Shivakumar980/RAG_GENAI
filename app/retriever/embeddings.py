import numpy as np
from typing import List, Dict, Any, Optional
from langchain_openai import OpenAIEmbeddings
from app.core.config import settings 
from dotenv import load_dotenv
import os


class EmbeddingManager:
    """
    Manages embeddings for the FAQ retriever system.
    Handles generating, caching, and processing of text embeddings.
    """
    
    def __init__(self, model_name: str = "text-embedding-ada-002", cache_enabled: bool = True):
    # Directly load API key from .env file
   
        load_dotenv()  # Explicitly load .env file
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("Missing OpenAI API key! Please check your .env file.")
            
        # Initialize with explicit API key
        self.embedding_model = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=api_key  # Explicitly pass API key
        )
        
        self.cache_enabled = cache_enabled
        self.embedding_cache = {}  # Simple in-memory cache
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get an embedding for a text string.
        Uses cache if enabled and available.
        
        Args:
            text: The text to embed
            
        Returns:
            A list of floats representing the embedding vector
        """
        if self.cache_enabled and text in self.embedding_cache:
            return self.embedding_cache[text]
        
        embedding = self.embedding_model.embed_query(text)
        
        if self.cache_enabled:
            self.embedding_cache[text] = embedding
            
        return embedding
    
    def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a batch of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        # Check cache first for each text
        if self.cache_enabled:
            embeddings = []
            texts_to_embed = []
            indices = []
            
            for i, text in enumerate(texts):
                if text in self.embedding_cache:
                    embeddings.append(self.embedding_cache[text])
                else:
                    texts_to_embed.append(text)
                    indices.append(i)
            
            # If there are uncached texts, embed them
            if texts_to_embed:
                new_embeddings = self.embedding_model.embed_documents(texts_to_embed)
                
                # Update cache and fill in the embeddings list
                for idx, embedding in zip(indices, new_embeddings):
                    self.embedding_cache[texts[idx]] = embedding
                    embeddings.insert(idx, embedding)
                    
            return embeddings
        else:
            # If cache is disabled, just embed everything
            return self.embedding_model.embed_documents(texts)
    
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity (0-1 where 1 is most similar)
        """
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        return dot_product / (norm1 * norm2)
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.embedding_cache = {}
    
    def get_cache_size(self) -> int:
        """
        Get the current size of the embedding cache.
        
        Returns:
            Number of cached embeddings
        """
        return len(self.embedding_cache)