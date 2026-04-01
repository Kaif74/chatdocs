# ChatDocs

A full-stack Retrieval-Augmented Generation demo that ingests LLM-friendly documentation from LangChain, CrewAI, and Expo, answers developer questions from those docs, and evaluates answer quality with automatic metrics plus human rubric scoring.

## Table of Contents

- [Overview](#overview)
- [Stack](#stack)
- [Project Structure](#project-structure)
- [Domain and Data](#domain-and-data)
- [RAG Design](#rag-design)
- [Evaluation](#evaluation)
- [Challenges and Lessons](#challenges-and-lessons)
- [Backend Setup](#backend-setup)
- [Frontend Setup](#frontend-setup)
- [Main Features](#main-features)
- [API Endpoints](#api-endpoints)
- [Deployment Notes](#deployment-notes)
- [Notes](#notes)

## Overview

This README is organized as a short technical report followed by setup and
deployment instructions.

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

## Domain and Data

This project targets a developer-assistant domain: answering practical questions
from framework documentation.

Current documentation sources:

- LangChain docs (`https://docs.langchain.com/llms-full.txt`)
- CrewAI docs (`https://docs.crewai.com/llms-full.txt`)
- Expo docs (`https://docs.expo.dev/llms-full.txt`)

Why this dataset:

- It is real-world technical documentation with dense concepts and APIs.
- It is updated frequently, which is a good stress test for ingestion refresh
  and safe sync behavior.
- It includes multiple frameworks, enabling source-filtered retrieval and
  cross-source evaluation.

Data characteristics:

- Input format is LLM-friendly text files.
- Content is cleaned to reduce markup/navigation artifacts before chunking.
- Source metadata is attached per chunk for filtering and retrieval scoring.

## RAG Design

The system follows a standard retrieve-then-generate architecture.

1. Ingestion

- Fetch each source document.
- Clean text (remove noisy markup fragments and normalize whitespace).
- Split text into overlapping chunks using `RecursiveCharacterTextSplitter`.
- Embed chunks with `sentence-transformers/all-MiniLM-L6-v2`.
- Store vectors and metadata in the configured vector store:
  - Local Chroma for development
  - Qdrant Cloud for production persistence and reliability

2. Chunking strategy

- Chunking is done by character-based recursive splitting to preserve local
  context while avoiding oversized chunks.
- Overlap is used to reduce boundary loss (important for definitions and
  step-by-step instructions that span chunk edges).
- Per-source deterministic chunk IDs are used so rebuild/sync behavior remains
  predictable.

3. Embedding model choice

- Model: `all-MiniLM-L6-v2`
- Reasoning:
  - Good quality/latency tradeoff for semantic retrieval.
  - Lightweight enough for local and low-cost deployment constraints.
  - Works consistently across Chroma and Qdrant backends.

4. Retrieval strategy

- Similarity search over top-k chunks (default limit from retriever path).
- Optional source filter (`langchain`, `crewai`, `expo`) to constrain search.
- Retrieved chunks are formatted into a compact context block and sent to the
  generator.

5. Generation strategy and failover

- Provider chain: Mistral, OpenRouter, then NVIDIA as fallback order.
- Timeout-based fast failover: if one provider is slow/unavailable, the system
  advances to the next provider.
- Prompting instructs the model to stay grounded in retrieved documentation and
  avoid hallucinating when context is insufficient.

## Evaluation

Evaluation runs a fixed benchmark question set and computes automatic metrics
per question, then aggregates summary statistics.

### Metrics

1. Cosine Similarity

- Compares semantic similarity between expected answer and generated answer.
- Both texts are embedded with the same embedding model used in retrieval.
- Formula:

  `cosine(a, b) = (a . b) / (||a|| * ||b||)`

- Range: `0` to `1` in this implementation context (higher is better).

2. Keyword Overlap F1

- Builds keyword sets from expected/generated texts after normalization and
  stopword filtering.
- Computes precision and recall over token overlap, then F1:

  `precision = |overlap| / |generated_keywords|`

  `recall = |overlap| / |expected_keywords|`

  `F1 = 2 * precision * recall / (precision + recall)`

- Captures lexical faithfulness to expected concepts.

3. Retrieval Hit Rate

- Binary per question: `1` if any retrieved chunk source matches the benchmark
  source label, else `0`.
- Final hit rate is the mean of all binary hits.

4. Human Rubric (UI-assisted)

- Optional reviewer scores stored in browser local storage:
  - Coherence
  - Completeness
  - Factual correctness

### Runtime

- Full evaluation runs all benchmark questions.
- Bounded parallelism via `EVAL_CONCURRENCY` (default `6`, capped in code).
- Each question performs full RAG execution (retrieve + generate + metrics), so
  runtime depends heavily on provider latency.
- `LLM_PROVIDER_TIMEOUT_SECONDS` prevents slow providers from blocking the whole
  run and improves end-to-end stability.

### Sample Results

Example response shape from `POST /eval/run`:

```json
{
  "summary": {
    "average_cosine_similarity": 0.81,
    "average_keyword_f1": 0.67,
    "retrieval_hit_rate": 0.91
  },
  "results": [
    {
      "id": 1,
      "source": "langchain",
      "provider_used": "mistral",
      "cosine_similarity": 0.88,
      "keyword_f1": 0.72,
      "retrieval_hit": 1
    }
  ]
}
```

Note: values vary by provider availability, upstream model behavior, and
documentation freshness at ingestion time.

## Challenges and Lessons

1. Stateful vectors vs ephemeral hosting

- Challenge: local-only vector persistence is fragile on serverless/ephemeral
  platforms.
- Resolution: support Qdrant Cloud and disable heavy startup ingestion in
  production (`AUTO_INGEST_ON_STARTUP=false`).
- Lesson: separate compute lifecycle from vector persistence for reliable deploys.

2. Long ingestion and operational visibility

- Challenge: large upserts can appear stalled and are hard to diagnose without
  progress reporting.
- Resolution: added ingestion progress endpoint and frontend progress polling.
- Lesson: long-running data operations need transparent progress surfaces.

3. Destructive rebuild risk

- Challenge: accidental full rebuild can wipe vectors and force lengthy reingest.
- Resolution: safe sync as default, explicit rebuild mode, warning banner, and
  confirmation before full rebuild.
- Lesson: make destructive operations explicit and hard to trigger accidentally.

4. Provider latency and reliability variance

- Challenge: some LLM providers are slow or intermittently unavailable.
- Resolution: timeout-based failover and configurable concurrency.
- Lesson: production RAG needs graceful degradation, not single-provider dependence.

5. Dataset alignment drift

- Challenge: benchmark items can drift from active ingested sources.
- Resolution: removed outdated source questions and aligned filters/UI.
- Lesson: keep ingestion set, benchmark set, and UI controls synchronized.

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
LLM_PROVIDER_TIMEOUT_SECONDS=15
EVAL_CONCURRENCY=6

VECTOR_STORE_PROVIDER=chroma
CHROMA_PERSIST_DIR=./chroma_db

QDRANT_URL=
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=docs
QDRANT_BATCH_SIZE=64
QDRANT_UPSERT_RETRIES=4
QDRANT_TIMEOUT_SECONDS=60

AUTO_INGEST_ON_STARTUP=true
```

For production reliability on low-memory hosts, prefer Qdrant Cloud over local Chroma persistence:

```env
VECTOR_STORE_PROVIDER=qdrant
QDRANT_URL=https://<your-cluster>.cloud.qdrant.io
QDRANT_API_KEY=<your-key>
QDRANT_COLLECTION_NAME=docs
AUTO_INGEST_ON_STARTUP=false
```

Recommended workflow:

- Keep `AUTO_INGEST_ON_STARTUP=true` locally for development.
- In production, set `AUTO_INGEST_ON_STARTUP=false`, then call `POST /ingest` once after deploy.

4. Start the backend:

```bash
uvicorn main:app --reload
```

The backend will check whether Chroma already has data on startup. If the collection is empty, it will automatically ingest all configured documentation sources.

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
- Evaluate tab: run all benchmark questions and review automatic metrics plus local human rubric scores
- Ingest tab: inspect chunk counts, run safe sync, and rebuild vectors only when required
- Empty-store protection: `/query` and `/eval/run` return a clear `503` with `No documents ingested yet` if the collection is empty

## API Endpoints

- `POST /ingest`
- `GET /ingest/status`
- `GET /ingest/progress`
- `GET /health`
- `POST /query`
- `POST /eval/run`
- `GET /eval/questions`

`POST /ingest` modes:

- Default safe mode: ingests only missing sources and keeps existing vectors.
- Full rebuild: `POST /ingest?rebuild=true` wipes and recreates the collection first.
- Refresh existing sources without full wipe: `POST /ingest?refresh_existing=true`.

Important: Full rebuild is destructive for the current vector collection and can take much longer. Prefer safe mode unless a full re-index is truly needed.

## Deployment Notes

- For Railway, set the backend root to `chatdocs/backend` and run:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

- Configure Railway healthchecks to call `GET /health`.
- For stable Railway deploys, use Qdrant Cloud by setting `VECTOR_STORE_PROVIDER=qdrant` and Qdrant env vars.
- If you stay on Chroma in Railway, use a persistent volume for `CHROMA_PERSIST_DIR` so stored chunks survive restarts.
- Set `AUTO_INGEST_ON_STARTUP=false` in Railway to avoid heavy cold-start ingestion.
- For Vercel, set the frontend root to `chatdocs/frontend` and set `VITE_API_BASE_URL` to your Railway backend URL.

## Notes

- Re-ingestion is idempotent and rebuilds the collection rather than duplicating chunks.
- Embeddings are local and use `all-MiniLM-L6-v2`.
- Human rubric scores are stored in browser `localStorage` and are not persisted on the backend.
- Full evaluation runs live every time and uses bounded parallelism for faster execution.
