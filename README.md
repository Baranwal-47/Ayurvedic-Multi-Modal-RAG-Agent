# Ayurveda Multimodal RAG

Grounded RAG system for Ayurveda content with text + image retrieval, reranking, and a developer inspection UI.

## Repo At A Glance

- `backend/`: FastAPI API, retrieval/rerank pipeline, Qdrant integration, ingestion scripts.
- `frontend/`: Next.js app with end-user chat view and `/developer` debug view.
- `backend/data/`: Local data, PDFs, ingestion artifacts.
- `backend/tests/`: Retrieval/query pipeline tests.

## Main Backend Modules

- `backend/api/main.py`: API entrypoint (`/health`, `/query`, `/query/stream`).
- `backend/retrieval/hybrid_search.py`: Hybrid retrieval + routing (`simple`, `fast`, `deep`).
- `backend/retrieval/reranker.py`: Cross-encoder reranker with prewarm and debug timings.
- `backend/rag/query_engine.py`: End-to-end orchestration.
- `backend/rag/context_builder.py`: Prompt/context assembly from reranked evidence.
- `backend/vector_db/qdrant_client.py`: Qdrant search and point retrieval helpers.

## Prerequisites

- Python 3.11+ (3.12 recommended)
- Node.js 18+
- Access to a Qdrant instance
- Groq API key for answer generation
- For ingestion/OCR/image workflows: Google Vision + Cloudinary credentials

## Backend Setup And Run (Windows PowerShell)

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

For contributors running tests/local checks, install dev dependencies too:

```powershell
pip install -r requirements-dev.txt
```

Fill required values in `.env` (at minimum: `QDRANT_URL`, `QDRANT_API_KEY`, `GROQ_API_KEY`).

Run API:

```powershell
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

## Frontend Setup And Run

```powershell
cd frontend
npm install
```

Create `frontend/.env.local`:

```bash
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

Run frontend:

```powershell
npm run dev
```

Open:

- App: `http://localhost:3000`
- Developer view: `http://localhost:3000/developer`

## API Quick Use

`POST /query`

```json
{
	"query": "What is Swedana Yantra?",
	"include_debug": true
}
```

`POST /query/stream` returns SSE tokens + final payload.

## Tests

Backend focused suite:

```powershell
Set-Location d:\Docs\Computer\ayurveda-rag
backend\venv\Scripts\python.exe -m pytest backend/tests/test_retrieval_query_api.py -q
```

Frontend lint:

```powershell
cd frontend
npm run lint
```

## Collaboration Notes

- `backend/requirements.txt` is runtime-only and pinned for reproducibility.
- `backend/requirements-dev.txt` includes runtime + test tooling.
- Avoid committing full `pip list` or `pip freeze` outputs; they include machine-specific transitive packages and often create conflicts for collaborators.

## Notes

- If you run `uvicorn ... --reload`, model warmup can appear multiple times because worker processes restart on file changes.
- For stable latency checks/manual performance testing, run without `--reload`.
