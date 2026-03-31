# ChatDocs

A full-stack Retrieval-Augmented Generation demo that ingests LLM-friendly documentation from LangChain, CrewAI, Next.js, and Expo, answers developer questions from those docs, and evaluates answer quality with automatic metrics plus human rubric scoring.

## Stack

- Backend: Python, FastAPI, LangChain, ChromaDB, sentence-transformers
- Frontend: React with Vite, native `fetch`
- LLM providers: NVIDIA NIM, Mistral API, and OpenRouter fallback chain
- Evaluation: cosine similarity, keyword overlap F1, retrieval hit rate, human rubric scoring

## Project Structure

```text
chatdocs/
|- backend/
|  |- main.py
|  |- rag/
|  |- eval/
|  |- chroma_db/
|  |- requirements.txt
|  `- .env.example
`- frontend/
   |- index.html
   |- package.json
   |- vite.config.js
   `- src/
```

## Backend Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
cd backend
pip install -r requirements.txt
```

3. Copy the example environment file and fill in your API keys:

```bash
cp .env.example .env
```

Set:

```env
OPENROUTER_API_KEY=
NVIDIA_API_KEY=
MISTRAL_API_KEY=
CHROMA_PERSIST_DIR=./chroma_db
```

4. Start the backend:

```bash
uvicorn main:app --reload
```

The backend will check whether Chroma already has data on startup. If the collection is empty, it will automatically ingest all four documentation sources.

Note: on the very first startup, `sentence-transformers/all-MiniLM-L6-v2` may need to be downloaded if it is not already cached locally. That initial model download can add roughly 1 to 2 minutes before the app is ready.

## Frontend Setup

1. Install dependencies:

```bash
cd frontend
npm install
```

2. Start the Vite dev server:

```bash
npm run dev
```

By default, the frontend expects the FastAPI backend at `http://localhost:8000`. You can override that with `VITE_API_BASE_URL`.

## Main Features

- Chat tab: ask a question, optionally filter by source, and inspect retrieved chunks
- Evaluate tab: run all 15 benchmark questions and review automatic metrics plus local human rubric scores
- Ingest tab: inspect chunk counts and rebuild the Chroma collection
- Empty-store protection: `/query` and `/eval/run` return a clear `503` with `No documents ingested yet` if the collection is empty

## API Endpoints

- `POST /ingest`
- `GET /ingest/status`
- `GET /health`
- `POST /query`
- `POST /eval/run`
- `GET /eval/questions`

## Deployment Notes

- For Railway, set the backend root to `chatdocs/backend` and run:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

- Configure Railway healthchecks to call `GET /health`.
- If you deploy Chroma on Railway, use a persistent volume for `CHROMA_PERSIST_DIR` so stored chunks survive restarts.
- For Vercel, set the frontend root to `chatdocs/frontend` and set `VITE_API_BASE_URL` to your Railway backend URL.

## Notes

- Re-ingestion is idempotent and rebuilds the collection rather than duplicating chunks.
- Embeddings are local and use `all-MiniLM-L6-v2`.
- Human rubric scores are stored in browser `localStorage` and are not persisted on the backend.
- Full evaluation runs live every time and uses bounded parallelism for faster execution.
