"""
Script to index FAQs and create FAISS index and metadata.

Usage:
    python -m scripts.index_faqs --faq-file path/to/faqs.json
"""

import json
import tiktoken
import numpy as np
import faiss
import os
import unicodedata
import argparse
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Add parent directory to path to import app modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings

def normalize_text(text: str) -> str:
    """
    Normalize text by converting special Unicode chars to standard form 
    and removing extra spaces.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    return unicodedata.normalize("NFKC", text).strip()

def process_faqs(faq_file: str) -> tuple:
    """
    Process FAQs and create chunks with metadata.
    
    Args:
        faq_file: Path to FAQ JSON file
        
    Returns:
        Tuple of (chunks, metadata)
    """
    # Load FAQs
    with open(faq_file, "r", encoding="utf-8") as f:
        faqs = json.load(f)
    
    print(f"Loaded {len(faqs)} FAQs from {faq_file}")
    
    # Apply normalization
    for faq in faqs:
        faq["question"] = normalize_text(faq["question"])
        faq["answer"] = normalize_text(faq["answer"])
    
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    # Define chunking strategy
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=20,
        separators=["\n\n", ".", " "],
    )
    
    # Process FAQs into chunks
    chunks = []
    metadata = []
    
    for faq in faqs:
        question = faq["question"]
        answer = faq["answer"]
        
        # Store question as its own entry
        chunks.append(question)
        metadata.append({"type": "question", "question": question})
        
        # Split answer into chunks if needed
        split_answers = text_splitter.split_text(answer)
        
        for part in split_answers:
            chunks.append(part)
            metadata.append({"type": "answer", "question": question, "chunk": part})
    
    print(f"Created {len(chunks)} chunks ({len(metadata)} metadata entries)")
    return chunks, metadata

def create_embeddings(chunks: List[str]) -> np.ndarray:
    """
    Generate embeddings for chunks.
    
    Args:
        chunks: List of text chunks
        
    Returns:
        NumPy array of embeddings
    """
    print("Generating embeddings...")
    embedding_model = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
    embeddings = embedding_model.embed_documents(chunks)
    
    # Convert to NumPy array
    embeddings_array = np.array(embeddings, dtype=np.float32)
    print(f"Generated embeddings with shape {embeddings_array.shape}")
    
    return embeddings_array

def save_index_and_metadata(embeddings_array: np.ndarray, metadata: List[Dict], 
                            faiss_index_file: str, metadata_file: str) -> None:
    """
    Save FAISS index and metadata to disk.
    
    Args:
        embeddings_array: NumPy array of embeddings
        metadata: List of metadata dictionaries
        faiss_index_file: Path to save FAISS index
        metadata_file: Path to save metadata
    """
    # Initialize FAISS index
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(faiss_index_file), exist_ok=True)
    os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
    
    # Save FAISS index and metadata
    faiss.write_index(index, faiss_index_file)
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    
    print(f"Saved FAISS index to {faiss_index_file}")
    print(f"Saved metadata to {metadata_file}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Index FAQs and create FAISS index and metadata")
    parser.add_argument("--faq-file", default="faqs.json", help="Path to FAQ JSON file")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OpenAI API key! Set OPENAI_API_KEY in environment variables or .env file.")
    
    # Process FAQs
    chunks, metadata = process_faqs(args.faq_file)
    
    # Generate embeddings
    embeddings_array = create_embeddings(chunks)
    
    # Save index and metadata
    save_index_and_metadata(
        embeddings_array, 
        metadata, 
        settings.FAISS_INDEX_FILE, 
        settings.METADATA_FILE
    )
    
    print(f"✅ Process completed! Files saved in:")
    print(f"  - FAISS Index: {settings.FAISS_INDEX_FILE}")
    print(f"  - Metadata: {settings.METADATA_FILE}")
    print(f"Metadata Preview: {metadata[:2]}")

if __name__ == "__main__":
    main()