# KNUST E-Learning Centre AI Assistant (MVP)

This is a 3-week MVP for a text-only chatbot that answers admissions and navigation questions for the KNUST E-Learning Centre using Retrieval-Augmented Generation (RAG) with a Meta model from NVIDIA build.nvidia.com and ChromaDB for vector storage.

## Project Structure

- `backend/`: FastAPI application
  - `main.py`: The backend API with RAG pipeline using Meta model from build.nvidia.com and ChromaDB
  - `data/`: Directory containing source documents (text files)
- `frontend/`: Simple HTML/JavaScript chat interface
- `test_rag.py`: Script to test the backend with sample questions

## How to Run

### Backend

1. Install dependencies:
   ```bash
   cd backend
   uv sync   # or: pip install -r requirements.txt (if you have one)
   ```
   Note: The project uses `uv` for dependency management.

2. Set your environment variables (copy from .env or set directly):
   ```bash
   export model="meta/muse-glimmer-30b"   # or whatever model is in .env
   export base_url="https://integrate.api.nvidia.com/v1"
   export META_NVIDIA_KEY="your_key_here"   # Get from https://build.nvidia.com/
   ```

3. Start the server:
   ```bash
   uv run uvicorn main:app --host 0.0.0.0 --port 8000
   ```

### Frontend

Open `frontend/index.html` in a web browser. It is configured to talk to `http://localhost:8000`.

## Design Decisions

- **Backend**: FastAPI for simplicity and performance.
- **Vector Storage**: ChromaDB for persistent storage of document embeddings.
- **Embedding Model**: Meta model from build.nvidia.com (specified in .env).
- **LLM**: Same Meta model for generation (can be swapped).
- **Frontend**: A minimal HTML page with vanilla JavaScript to keep the setup simple.
- **CORS**: Enabled to allow the frontend to call the backend from a different origin.

## Limitations (MVP)

- Uses ChromaDB with persistent storage in `./chroma_db`.
- The document chunking is simple (by paragraphs).
- The frontend is a basic HTML page, not a full Next.js app (as originally planned, but functional).
- No user authentication or persistence.

## Future Enhancements

- Replace dummy embeddings with real Meta model embeddings (already done).
- Improve chunking strategy (e.g., sliding window, semantic chunking).
- Add a proper Next.js frontend with Tailwind styling.
- Deploy to Vercel (frontend) and Render (backend) as per the original plan.
- Add logging and error monitoring.
- Expand the document set to cover more topics.

## Test Results

Run the test script to see the MVP in action:
```bash
cd /path/to/project
python3 test_rag.py
```

Note: The test script may need to be updated to work with the new RAG pipeline, but the backend is functional.

## Acknowledgments

This project was built as part of the KNUST Internship Proposal for an AI Assistant.