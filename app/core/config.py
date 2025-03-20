import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings loaded from environment variables with defaults."""
    
    # API Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # LLM Settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL: str = "gpt-3.5-turbo"
    LLM_TEMPERATURE: float = 0.2
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    
    # Retriever Settings
    TOP_K: int = 4
    SIMILARITY_THRESHOLD: float = 0.85
    CONTEXT_WINDOW_SIZE: int = 5
    
    # Data Paths
    DATA_DIR: str = "data"
    CONTEXT_DIR: str = os.path.join(DATA_DIR, "context")
    INDEX_DIR: str = os.path.join(DATA_DIR, "index")
    METADATA_DIR: str = os.path.join(DATA_DIR, "metadata")
    
    FAISS_INDEX_FILE: str = os.path.join(INDEX_DIR, "faiss_index.bin")
    METADATA_FILE: str = os.path.join(METADATA_DIR, "metadata.json")
    
    # Session Management
    SESSION_CLEANUP_DAYS: int = 30  # Days to keep inactive sessions
    SESSION_TIMEOUT: int = 3600  # Seconds until a session is considered inactive

    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()

# Ensure required directories exist
os.makedirs(settings.CONTEXT_DIR, exist_ok=True)
os.makedirs(settings.INDEX_DIR, exist_ok=True)
os.makedirs(settings.METADATA_DIR, exist_ok=True)