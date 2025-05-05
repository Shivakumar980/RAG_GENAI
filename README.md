# RAG_GENAI: Conversational RAG Chatbot

A Retrieval Augmented Generation (RAG) Chatbot designed to answer user questions
## Overview

This application implements a conversational chatbot that uses the RAG (Retrieval Augmented Generation) approach to answer questions about FSU student finance. The system maintains conversation context to provide more relevant follow-up answers, uses a FAISS vector index for efficient similarity search, and includes persistence for session management.

## Features

- **Conversation Context**: Maintains conversation history for more coherent responses to follow-up questions
- **Vector Search**: Uses FAISS for fast semantic similarity search
- **Session Management**: Supports multiple conversation sessions with persistence
- **OpenAI Integration**: Leverages GPT models for response generation and embeddings
- **RESTful API**: Provides a clean API for integration with any frontend

## Prerequisites

- Python 3.8+
- OpenAI API key
- FastAPI
- FAISS vector database

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/RAG_GENAI.git
   cd RAG_GENAI
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Setting Up Data Directories

Create the necessary directories for storing application data:

```bash
mkdir -p data/context data/index data/metadata
```

## Indexing the FAQs

Before running the application, you need to index the FAQ data:

```bash
python -m scripts.index_faqs --faq-file faqs.json
```

This script:
- Loads FAQs from `faqs.json`
- Normalizes the text
- Chunks the answers into manageable pieces
- Generates embeddings using OpenAI's embedding model
- Creates a FAISS index for efficient similarity search
- Saves the index and metadata for later use

## Running the Application

Start the FastAPI server:

```bash
python main.py
```

By default, the server will run at `http://0.0.0.0:8000`.

You can access the API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### Query Endpoint

- `POST /api/query`: Submit a query to the FAQ chatbot
  - Request body: `{"query": "Your question here", "session_id": "optional_session_id"}`
  - Response: Answer with metadata and session information

### Session Management

- `GET /api/sessions`: List all active sessions
- `POST /api/sessions`: Create a new session
- `DELETE /api/sessions/{session_id}`: Delete a session
- `POST /api/sessions/{session_id}/clear`: Clear a session's conversation history
- `GET /api/sessions/{session_id}/history`: Get conversation history for a session
  - Query parameter: `full_history=true|false` (default: false)

### Maintenance

- `POST /api/maintenance`: Run maintenance tasks (clean up old sessions)
  - Query parameter: `days` (optional, number of days to keep sessions)

## Example API Usage

### Submit a query:

```bash
curl -X POST "http://localhost:8000/api/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "When is tuition due?", "session_id": null}'
```

### Create a new session:

```bash
curl -X POST "http://localhost:8000/api/sessions"
```

### Get session history:

```bash
curl -X GET "http://localhost:8000/api/sessions/{session_id}/history"
```

## Configuration

Application settings can be configured in `app/core/config.py`. Key settings include:

- `LLM_MODEL`: OpenAI model for response generation
- `EMBEDDING_MODEL`: OpenAI model for embeddings
- `TOP_K`: Number of related items to retrieve
- `SIMILARITY_THRESHOLD`: Threshold for direct matches
- `CONTEXT_WINDOW_SIZE`: Maximum conversation history to maintain
- `SESSION_CLEANUP_DAYS`: Days to keep inactive sessions

## Testing

Run the tests using:

```bash
python -m unittest discover -s test
```

## Project Structure

- `app/`: Main application code
  - `api/`: API endpoints and models
  - `context/`: Context window management for conversations
  - `core/`: Core application configuration
  - `retriever/`: FAQ retrieval and embedding logic
  - `utils/`: Utility functions for session management
- `data/`: Data storage (created by the application)
  - `context/`: Conversation context storage
  - `index/`: FAISS index storage
  - `metadata/`: Metadata storage
- `scripts/`: Utility scripts
  - `index_faqs.py`: Script for indexing FAQs
- `test/`: Test cases
- `faqs.json`: Source FAQ data
- `main.py`: Application entry point
- `requirements.txt`: Project dependencies

## Troubleshooting

### Common Issues

1. **OpenAI API Key**: Ensure your API key is correctly set in the `.env` file
2. **Missing Directories**: Verify that all required data directories exist
3. **FAISS Index**: If search doesn't work, make sure the index was created properly
4. **Python Version**: Ensure you're using Python 3.8 or higher

### Debug Logging

To enable debug logging, set `DEBUG=True` in `app/core/config.py`.

## License

[Your License Here]

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -am 'Add my feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Submit a pull request
