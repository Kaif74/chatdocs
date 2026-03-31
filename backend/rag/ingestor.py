import logging
import os
import re
from functools import lru_cache
from typing import Any

import chromadb
import httpx
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

# This project uses sentence-transformers on the PyTorch path only.
# Forcing Transformers to skip TensorFlow avoids local Keras/TensorFlow conflicts.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

from sentence_transformers import SentenceTransformer

load_dotenv()

logger = logging.getLogger(__name__)

COLLECTION_NAME = "docs"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DOC_SOURCES = {
    "langchain": "https://docs.langchain.com/llms-full.txt",
    "crewai": "https://docs.crewai.com/llms-full.txt",
    "nextjs": "https://nextjs.org/llms-full.txt",
    "expo": "https://docs.expo.dev/llms-full.txt",
}
DEFAULT_CHROMA_BATCH_SIZE = 1000


class NoDocumentsIngestedError(RuntimeError):
    """Raised when the vector store is empty."""


class MiniLMEmbeddingFunction(EmbeddingFunction[Documents]):
    """Chroma-compatible embedding adapter backed by sentence-transformers."""

    def __call__(self, input: Documents) -> Embeddings:
        model = get_embedding_model()
        vectors = model.encode(list(input), normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)


@lru_cache(maxsize=1)
def get_embedding_function() -> MiniLMEmbeddingFunction:
    return MiniLMEmbeddingFunction()


def get_persist_directory() -> str:
    return os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.PersistentClient:
    persist_dir = get_persist_directory()
    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)


def get_collection(recreate: bool = False):
    client = get_chroma_client()

    if recreate:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            logger.debug("Collection %s did not exist during rebuild.", COLLECTION_NAME)

        return client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=get_embedding_function(),
            metadata={"hnsw:space": "cosine"},
        )

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def get_collection_count() -> int:
    return get_collection().count()


def ensure_documents_ingested() -> None:
    if get_collection_count() == 0:
        raise NoDocumentsIngestedError("No documents ingested yet")


def build_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""],
    )


def _clean_document_text(text: str) -> str:
    cleaned_lines: list[str] = []

    for raw_line in text.replace("\r\n", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            cleaned_lines.append("")
            continue

        heading_match = re.match(r"^<h([1-6])>(.*?)</h\1>$", line, flags=re.IGNORECASE)
        if heading_match:
            level = int(heading_match.group(1))
            heading_text = re.sub(r"<[^>]+>", "", heading_match.group(2)).strip()
            cleaned_lines.append(f"{'#' * level} {heading_text}")
            continue

        if re.match(r"^</?[A-Za-z][^>]*>$", line):
            continue

        line = re.sub(r"</?code>", "`", line, flags=re.IGNORECASE)
        line = re.sub(r"</?strong>", "**", line, flags=re.IGNORECASE)
        line = re.sub(r"</?em>", "*", line, flags=re.IGNORECASE)
        line = re.sub(r"<br\s*/?>", " ", line, flags=re.IGNORECASE)
        line = re.sub(r"<[^>]+>", "", line)
        line = re.sub(r"\s+", " ", line).strip()

        if not line:
            continue

        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    return cleaned_text.strip()


async def fetch_source_text(source: str, url: str) -> str | None:
    try:
        async with httpx.AsyncClient(timeout=45.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.text
    except Exception as exc:
        logger.warning("Skipping %s because fetch failed: %s", source, exc)
        return None


def _source_count(collection, source: str) -> int:
    result = collection.get(where={"source": source}, include=["metadatas"])
    return len(result.get("ids", []))


def _get_chroma_batch_size(collection) -> int:
    max_batch_size = getattr(collection._client, "get_max_batch_size", None)
    if callable(max_batch_size):
        try:
            return min(DEFAULT_CHROMA_BATCH_SIZE, int(max_batch_size()))
        except Exception:
            logger.debug("Could not read Chroma max batch size, using default.")
    return DEFAULT_CHROMA_BATCH_SIZE


def _add_chunks_in_batches(
    collection,
    ids: list[str],
    chunks: list[str],
    metadatas: list[dict[str, Any]],
) -> None:
    batch_size = _get_chroma_batch_size(collection)
    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=chunks[start:end],
            metadatas=metadatas[start:end],
        )


def get_ingest_status() -> dict[str, Any]:
    collection = get_collection()
    total_chunks = collection.count()
    counts = {source: _source_count(collection, source) for source in DOC_SOURCES}

    return {
        "ready": total_chunks > 0,
        "total_chunks": total_chunks,
        "counts_by_source": counts,
    }


async def ingest_all_sources() -> dict[str, Any]:
    splitter = build_splitter()
    collection = get_collection(recreate=True)

    total_chunks = 0
    skipped_sources: list[dict[str, str]] = []
    counts_by_source = {source: 0 for source in DOC_SOURCES}

    for source, url in DOC_SOURCES.items():
        text = await fetch_source_text(source, url)
        if not text:
            skipped_sources.append(
                {"source": source, "url": url, "reason": "Failed to fetch source"}
            )
            continue

        text = _clean_document_text(text)

        chunks = splitter.split_text(text)
        if not chunks:
            skipped_sources.append(
                {"source": source, "url": url, "reason": "Source returned no text"}
            )
            continue

        ids = [f"{source}-{index}" for index in range(len(chunks))]
        metadatas = [
            {"source": source, "chunk_id": index, "url": url}
            for index in range(len(chunks))
        ]

        _add_chunks_in_batches(collection, ids, chunks, metadatas)
        counts_by_source[source] = len(chunks)
        total_chunks += len(chunks)
        logger.info("Ingested %s chunks for %s", len(chunks), source)

    return {
        "status": "completed",
        "ready": total_chunks > 0,
        "total_chunks": total_chunks,
        "counts_by_source": counts_by_source,
        "skipped_sources": skipped_sources,
    }
