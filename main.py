"""
Main entry point for the FAQ Retrieval API application.
"""

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import time
import os

from app.api.router import router
from app.core.config import settings

# Initialize FastAPI app
app = FastAPI(
    title="FAQ Retriever API",
    description="Retrieval-Augmented Generation API with conversation context",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include router
app.include_router(router)

# Add middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Error handler
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"An error occurred: {str(exc)}"}
    )

@app.get("/")
async def root():
    return {
        "message": "FAQ Retriever API",
        "docs": "/docs",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    # Check if required files exist
    faiss_exists = os.path.exists(settings.FAISS_INDEX_FILE)
    metadata_exists = os.path.exists(settings.METADATA_FILE)
    
    return {
        "status": "healthy" if faiss_exists and metadata_exists else "unhealthy",
        "checks": {
            "faiss_index": faiss_exists,
            "metadata": metadata_exists
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )