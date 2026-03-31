import logging
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

logger = logging.getLogger(__name__)

PROVIDERS = [
    {
        "name": "nvidia",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key_env": "NVIDIA_API_KEY",
        "model": "mistralai/mistral-large-3-675b-instruct-2512",
    },
    {
        "name": "mistral",
        "base_url": "https://api.mistral.ai/v1",
        "api_key_env": "MISTRAL_API_KEY",
        "model": "mistral-large-latest",
    },
    {
        "name": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model": "qwen/qwen3-next-80b-a3b-instruct:free",
    },
]


class LLMProviderError(RuntimeError):
    """Raised when all configured providers fail."""


def _is_temporary_provider_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)

    message = str(exc).lower()
    return status_code in {429, 503} or "rate limit" in message or "503" in message


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "\n".join(part.strip() for part in parts if part).strip()

    return str(content).strip()


async def generate_answer(question: str, context: str) -> dict[str, str]:
    errors: list[str] = []

    system_prompt = (
        "You are a documentation QA assistant for software developers. "
        "Answer only from the supplied documentation context. "
        "If the context is missing or insufficient, say that clearly instead of guessing. "
        "Prefer concise, technical answers. "
        "Ignore raw markup or navigation fragments if they appear in the context. "
        "Do not repeat the user's question. "
        "Do not say 'based on the provided documentation' unless the user asks. "
        "Use short paragraphs or bullets and focus on the actionable answer."
    )
    human_prompt = f"Question:\n{question}\n\nDocumentation context:\n{context}"

    for provider in PROVIDERS:
        api_key = os.getenv(provider["api_key_env"], "").strip()
        if not api_key:
            errors.append(
                f"{provider['name']}: missing environment variable {provider['api_key_env']}"
            )
            continue

        try:
            llm = ChatOpenAI(
                model=provider["model"],
                api_key=api_key,
                base_url=provider["base_url"],
                temperature=0.1,
                max_retries=0,
            )
            response = await llm.ainvoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
            )
            return {
                "answer": _normalize_content(response.content),
                "provider_used": provider["name"],
            }
        except Exception as exc:
            logger.warning("Provider %s failed: %s", provider["name"], exc)
            errors.append(f"{provider['name']}: {exc}")
            if _is_temporary_provider_error(exc):
                continue

    raise LLMProviderError(
        "All LLM providers failed. Check API keys, quotas, or upstream availability. "
        + " | ".join(errors)
    )
