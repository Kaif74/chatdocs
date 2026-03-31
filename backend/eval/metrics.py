import math
import re

from rag.ingestor import get_embedding_model

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "what",
    "when",
    "while",
    "with",
}


def _dot_product(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def cosine_similarity_score(expected_answer: str, generated_answer: str) -> float:
    model = get_embedding_model()
    expected_vector, generated_vector = model.encode(
        [expected_answer, generated_answer], normalize_embeddings=False
    )

    numerator = _dot_product(expected_vector.tolist(), generated_vector.tolist())
    denominator = _norm(expected_vector.tolist()) * _norm(generated_vector.tolist())
    if denominator == 0:
        return 0.0
    return round(max(0.0, min(numerator / denominator, 1.0)), 4)


def _keyword_set(text: str) -> set[str]:
    tokens = re.findall(r"[A-Za-z0-9_.-]+", text.lower())
    return {token for token in tokens if token not in STOPWORDS and len(token) > 1}


def keyword_overlap_f1(expected_answer: str, generated_answer: str) -> float:
    expected_keywords = _keyword_set(expected_answer)
    generated_keywords = _keyword_set(generated_answer)

    if not expected_keywords or not generated_keywords:
        return 0.0

    overlap = expected_keywords & generated_keywords
    precision = len(overlap) / len(generated_keywords)
    recall = len(overlap) / len(expected_keywords)
    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


def retrieval_relevance_score(
    expected_source: str, retrieved_chunks: list[dict[str, object]]
) -> int:
    return int(any(chunk.get("source") == expected_source for chunk in retrieved_chunks))
