# KNUST E-Learning Centre AI Assistant (MVP)

This is a 3-week MVP for a text-only chatbot that answers admissions and navigation questions for the KNUST E-Learning Centre using Retrieval-Augmented Generation (RAG) with a Meta model from NVIDIA build.nvidia.com and ChromaDB for vector storage.

## Features

- **Admissions Information**: Answers questions about undergraduate/graduate admissions, requirements, fees, scholarships, and procedures.
- **Campus Navigation**: Provides guidance on locating faculties, departments, halls, libraries, and key buildings on KNUST campus.
- **Multimodal Input**: Supports text, voice, image, and file inputs (voice and file processing require backend integration).
- **Retrieval-Augmented Generation**: Uses ChromaDB for efficient similarity search over admissions and navigation documents.
- **Meta Model Integration**: Leverages the NVIDIA Nemotron-3-Super model via the NVIDIA NIM API for embeddings and language generation.
- **Fallback Mechanism**: Gracefully degrades to dummy responses if API credentials are missing or invalid.
- **Responsive Frontend**: Clean, accessible chat interface with KNUST-inspired styling (blue and white theme).

## Tech Stack

- **Backend**: FastAPI, Uvicorn, Python 3.12
- **AI/ML**: 
  - NVIDIA Nemotron-3-Super (via NVIDIA NIM API)
  - ChromaDB (vector storage)
  - Sentence Transformers (fallback embedding model)
  - scikit-learn (fallback similarity search)
- **Frontend**: HTML5, CSS3, Vanilla JavaScript (Web Speech API for voice input)
- **DevOps**: 
  - uv (dependency management)
  - Git/GitHub (version control)
  - Environment variables (.env) for configuration

## Project Structure

```
KNUST-Intern-Proj/
├── backend/
│   ├── main.py          # FastAPI application entry point
│   ├── pyproject.toml   # Dependencies and project metadata
│   └── __pycache__/     # Python cache
├── frontend/
│   └── index.html       # Chat interface with voice/image/file inputs
├── data/
│   ├── admissions_faq.txt   # Sample admissions information
│   └── navigation_guide.txt # Campus navigation guide
├── test_rag.py          # Test script for RAG pipeline
├── README.md            # This file
└── .env                 # Environment variables (not tracked by Git)
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com:NeonTechno/KNUST-RAG-GER.git
cd KNUST-Intern-Proj
```

### 2. Install Dependencies
The project uses `uv` for fast, reliable dependency management.
```bash
# Install backend dependencies
cd backend
uv pip install -e .

# Install frontend dependencies (none required - vanilla JS/HTML/CSS)
cd ..
```

### 3. Configure Environment Variables
Create a `.env` file in the project root with the following variables:
```bash
# Model configuration (from .env)
model="meta/muse-glimmer-30b"          # or your desired NVIDIA model
base_url="https://integrate.api.nvidia.com/v1"
META_NVIDIA_KEY="nvapi-your_actual_key_here"

# Optional: Adjust these for different behavior
# TOP_K=3
# EMBEDDING_BATCH_SIZE=32
```

> **Note**: Get your NVIDIA API key from [build.nvidia.com](https://build.nvidia.com/) after selecting the Nemotron-3-Super model.

### 4. Run the Backend
```bash
cd backend
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

The backend will be available at `http://localhost:8000`.

### 5. Use the Frontend
Open `frontend/index.html` in a modern web browser (Chrome, Firefox, Safari, Edge).

## API Endpoints

### Health Check
```bash
GET http://localhost:8000/health
```
Returns `{"status": "ok"}` if the service is running.

### Query Processing
```bash
POST http://localhost:8000/query
Content-Type: application/json

{
  "question": "What are the admission requirements for undergraduate programs?",
  "top_k": 3
}
```

Returns:
```json
{
  "answer": "Generated answer based on retrieved context...",
  "sources": ["admissions_faq.txt:5", "navigation_guide.txt:12"]
}
```

## Document Preparation

The RAG pipeline loads text files from the `data/` directory at startup. To update the knowledge base:

1. Add `.txt` files to the `data/` directory (one document per file is recommended).
2. Restart the backend to reload and re-embed the documents.
3. Supported formats: Plain text (`.env`, `.txt`, `.md` will be treated as text).

## Design & Customization

### Frontend Styling
- **Primary Color**: `#003366` (KNUST blue)
- **Secondary Colors**: `#28a745` (voice), `#ffc107` (image), `#6f42c1` (file)
- **Typography**: Arial, sans-serif (system fallback)
- **Spacing**: Consistent 10-20px padding and margins
- **Border Radius**: 20-25px for soft, modern feel
- **Shadows**: Subtle elevation for depth

### Backend Configuration
Adjust these in `backend/main.py` or via environment variables:
- `TOP_K`: Number of retrieved chunks (default: 3)
- `CHROMA_PATH`: ChromaDB storage location (default: `./chroma_db`)
- `EMBEDDING_MODEL`: Fallback embedding model (default: `all-MiniLM-L6-v2`)

## How It Works (RAG Pipeline)

1. **Document Loading**: On startup, loads all `.txt` files from `data/`.
2. **Chunking**: Splits documents into ~500-word chunks with overlap.
3. **Embedding**: 
   - Primary: Uses NVIDIA NIM API with model from `.env`
   - Fallback: Uses `all-MiniLM-L6-v2` from sentence-transformers
4. **Storage**: Stores embeddings in ChromaDB for efficient similarity search.
5. **Query Processing**: 
   - Embeds user question using same model
   - Retrieves top-k similar chunks from ChromaDB
   - Constructs prompt with retrieved context
   - Generates answer using NVIDIA NIM chat completion
6. **Response**: Returns answer with source citations.

## Verification

### Health Check
```bash
curl -s http://localhost:8000/health
# Expected: {"status":"ok"}
```

### Sample Query
```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Where is the College of Engineering located?", "top_k": 2}'
```

### Frontend Testing
1. Open `frontend/index.html` in browser
2. Click microphone icon to test voice input (requires microphone permission)
3. Click image/upload icons to test file selection
4. Type questions and click Send or press Enter

## Next Steps (Post-MVP)

1. **Backend Enhancements**
   - Add file processing for PDFs, DOCs, images (OCR, image understanding)
   - Implement conversation history and context retention
   - Add rate limiting and API key validation
   - Create Dockerfile for containerized deployment
   - Add logging and monitoring (Prometheus/Grafana)

2. **Frontend Improvements**
   - Add message reactions and copy-to-clipboard
   - Implement typing indicators and read receipts
   - Add dark/light theme toggle
   - Create mobile-native version (React Native/Ionic)
   - Add conversation export/share functionality

3. **Knowledge Base Expansion**
   - Scrape official KNUST website for up-to-date information
   - Add video content with transcripts (YouTube channel)
   - Incorporate student handbook and academic calendar
   - Add multilingual support (Twi, French)

4. **Deployment**
   - Deploy to cloud platform (AWS, Azure, GCP)
   - Set up CI/CD pipeline with GitHub Actions
   - Implement HTTPS with SSL/TLS certificates
   - Add load balancing and auto-scaling

## Troubleshooting

### Common Issues

**1. Backend fails to start**
- Check `.env` file exists and contains required variables
- Verify `uv` is installed: `uv --version`
- Ensure port 8000 is free: `lsof -i :8000`

**2. API key errors**
- Verify NVIDIA API key is valid and has model access
- Check `base_url` is correct for your region
- Ensure network allows outbound connections to `integrate.api.nvidia.com`

**3. No responses or dummy responses**
- Backend will show warning: "META_NVIDIA_KEY not set. Using dummy responses."
- Confirm `.env` is in project root (same level as README)
- Restart backend after updating `.env`

**4. Frontend issues**
- Voice input requires HTTPS or localhost (for security)
- Image/file inputs show notifications; actual processing needs backend integration
- Clear browser cache if styles don't update

**5. ChromaDB errors**
- Delete `./chroma_db` directory to reset vector store
- Ensure write permissions in backend directory
- Check available disk space (ChromaDB persists to disk)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [NVIDIA](https://www.nvidia.com/) for providing access to the Nemotron-3-Super model via build.nvidia.com
- [ChromaDB](https://www.trychroma.com/) for the open-source vector database
- [FastAPI](https://fastapi.tiangolo.com/) for the high-performance Python web framework
- The KNUST E-Learning Centre for inspiring this project

---

**Ready to assist with admissions and navigation questions for the KNUST E-Learning Centre!**# Deployment test
