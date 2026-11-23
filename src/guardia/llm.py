"""Fábrica da LLM: provedor e modelo vêm do .env (LLM_PROVIDER, LLM_MODEL e a chave)."""
import logging
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

log = logging.getLogger(__name__)

PROVIDER_KEYS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "xai": "XAI_API_KEY",
    "groq": "GROQ_API_KEY",
}


def _config() -> tuple[str, str]:
    load_dotenv()
    return os.getenv("LLM_PROVIDER", "").strip(), os.getenv("LLM_MODEL", "").strip()


def get_llm() -> BaseChatModel | None:
    """Retorna a LLM configurada ou None (modo degradado: app roda sem texto da LLM)."""
    provider, model = _config()
    if not provider or not model:
        return None
    if provider not in PROVIDER_KEYS:
        log.warning("LLM_PROVIDER desconhecido: %s (aceitos: %s)", provider, ", ".join(PROVIDER_KEYS))
        return None
    if not os.getenv(PROVIDER_KEYS[provider]):
        log.warning("LLM_PROVIDER=%s sem a chave %s", provider, PROVIDER_KEYS[provider])
        return None
    return init_chat_model(model, model_provider=provider, temperature=0.2, timeout=60)


def llm_label() -> str:
    provider, model = _config()
    return f"{provider}:{model}" if get_llm() is not None else "desligada"
