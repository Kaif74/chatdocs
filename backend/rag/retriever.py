from typing import Any

from rag.ingestor import ensure_documents_ingested, get_collection


def _distance_to_score(distance: float) -> float:
    score = 1.0 - float(distance)
    return round(max(0.0, min(score, 1.0)), 4)


def retrieve_chunks(
    query: str,
    source_filter: str | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    ensure_documents_ingested()
    collection = get_collection()

    query_kwargs: dict[str, Any] = {
        "query_texts": [query],
        "n_results": limit,
        "include": ["documents", "metadatas", "distances"],
    }
    if source_filter:
        query_kwargs["where"] = {"source": source_filter}

    results = collection.query(**query_kwargs)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    formatted_results: list[dict[str, Any]] = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        formatted_results.append(
            {
                "text": document,
                "source": metadata.get("source"),
                "chunk_id": metadata.get("chunk_id"),
                "url": metadata.get("url"),
                "score": _distance_to_score(distance),
            }
        )

    return formatted_results
