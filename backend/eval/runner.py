import asyncio
from statistics import mean
from typing import Any

from eval.metrics import (
    cosine_similarity_score,
    keyword_overlap_f1,
    retrieval_relevance_score,
)
from eval.questions import QA_PAIRS
from rag.ingestor import ensure_documents_ingested
from rag.pipeline import run_rag_pipeline

EVAL_CONCURRENCY = 4


async def _evaluate_question(
    pair: dict[str, Any], semaphore: asyncio.Semaphore
) -> dict[str, Any]:
    async with semaphore:
        pipeline_result = await run_rag_pipeline(pair["question"])

    cosine_score = cosine_similarity_score(
        pair["expected_answer"], pipeline_result["answer"]
    )
    keyword_f1 = keyword_overlap_f1(
        pair["expected_answer"], pipeline_result["answer"]
    )
    retrieval_hit = retrieval_relevance_score(pair["source"], pipeline_result["sources"])

    return {
        "id": pair["id"],
        "source": pair["source"],
        "question": pair["question"],
        "expected_answer": pair["expected_answer"],
        "generated_answer": pipeline_result["answer"],
        "provider_used": pipeline_result["provider_used"],
        "sources": pipeline_result["sources"],
        "cosine_similarity": cosine_score,
        "keyword_f1": keyword_f1,
        "retrieval_hit": retrieval_hit,
        "rubric": {
            "coherence": None,
            "completeness": None,
            "factual_correctness": None,
        },
    }


async def run_full_evaluation() -> dict[str, Any]:
    ensure_documents_ingested()

    semaphore = asyncio.Semaphore(EVAL_CONCURRENCY)
    tasks = [
        _evaluate_question(question_pair, semaphore) for question_pair in QA_PAIRS
    ]
    results = await asyncio.gather(*tasks)

    summary = {
        "average_cosine_similarity": round(
            mean(result["cosine_similarity"] for result in results), 4
        ),
        "average_keyword_f1": round(mean(result["keyword_f1"] for result in results), 4),
        "retrieval_hit_rate": round(
            mean(result["retrieval_hit"] for result in results), 4
        ),
    }

    return {"summary": summary, "results": results}
