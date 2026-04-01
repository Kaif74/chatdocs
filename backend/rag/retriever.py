from typing import Any

from rag.ingestor import search_documents


def retrieve_chunks(
    query: str,
    source_filter: str | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    return search_documents(query, source_filter=source_filter, limit=limit)
