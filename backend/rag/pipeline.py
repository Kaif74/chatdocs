from typing import Any

from rag.generator import generate_answer
from rag.retriever import retrieve_chunks


def _build_context(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return (
            "No relevant documentation chunks were retrieved for this question. "
            "If the answer cannot be grounded in the available docs, say so clearly."
        )

    parts = []
    for chunk in chunks:
        parts.append(
            "\n".join(
                [
                    f"Source: {chunk['source']}",
                    f"Chunk ID: {chunk['chunk_id']}",
                    f"Score: {chunk['score']}",
                    chunk["text"],
                ]
            )
        )
    return "\n\n---\n\n".join(parts)


async def run_rag_pipeline(
    question: str, source_filter: str | None = None
) -> dict[str, Any]:
    sources = retrieve_chunks(question, source_filter=source_filter)
    context = _build_context(sources)
    generation = await generate_answer(question, context)

    return {
        "answer": generation["answer"],
        "sources": sources,
        "provider_used": generation["provider_used"],
    }
