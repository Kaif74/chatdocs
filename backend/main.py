import logging
from contextlib import asynccontextmanager
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from eval.questions import QA_PAIRS
from eval.runner import run_full_evaluation
from rag.generator import LLMProviderError
from rag.ingestor import (
    NoDocumentsIngestedError,
    get_collection_count,
    get_ingest_status,
    ingest_all_sources,
)
from rag.pipeline import run_rag_pipeline

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    source_filter: Literal["langchain", "crewai", "nextjs", "expo"] | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    if get_collection_count() == 0:
        logger.info("Chroma collection is empty, starting initial ingestion.")
        try:
            await ingest_all_sources()
        except Exception as exc:
            logger.warning("Initial ingestion failed: %s", exc)
    else:
        logger.info("Existing Chroma collection detected, skipping startup ingestion.")
    yield


app = FastAPI(title="ChatDocs", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ingest")
async def ingest_documents():
    return await ingest_all_sources()


@app.get("/ingest/status")
async def ingest_status():
    return get_ingest_status()


@app.get("/health")
async def healthcheck():
    status = get_ingest_status()
    return {
        "status": "ok",
        "ready": status["ready"],
        "total_chunks": status["total_chunks"],
    }


@app.post("/query")
async def query_documents(payload: QueryRequest):
    try:
        return await run_rag_pipeline(
            payload.question, source_filter=payload.source_filter
        )
    except NoDocumentsIngestedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc


@app.post("/eval/run")
async def run_evaluation():
    try:
        return await run_full_evaluation()
    except NoDocumentsIngestedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Evaluation failed")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}") from exc


@app.get("/eval/questions")
async def get_questions():
    return {"questions": QA_PAIRS}
