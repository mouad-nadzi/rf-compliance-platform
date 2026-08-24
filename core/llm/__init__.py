"""
core/llm — Pluggable LLM Engine Package & Unified Generation Facade.

Contains concrete LLM engine implementations (e.g. QwenAgentWorld35BEngine)
and exposes a clean, model-agnostic `generate_json(system_prompt, user_prompt)`
function. Each engine owns its native prompt template and think/no-think switch
via the BaseLLMEngine hook (_generate_raw).
"""

import logging
from core.registry import get_llm_engine

logger = logging.getLogger(__name__)

# Singleton LLM instance
_engine_instance = None


def _get_engine():
    """Get or load the configured LLM engine singleton."""
    global _engine_instance
    if _engine_instance is None:
        from server.config import LLM_ENGINE, CACHE_DIR
        logger.info(f"LLM facade: initializing engine '{LLM_ENGINE}' via registry...")
        _engine_instance = get_llm_engine(LLM_ENGINE)
        _engine_instance.load(CACHE_DIR)
    return _engine_instance


def load_llm_engine():
    """Eagerly load the configured LLM engine into GPU memory on API startup."""
    _get_engine()


from server.config import DEFAULT_MAX_TOKENS

def generate_json(
    system_prompt: str,
    user_prompt: str,
    disable_thinking: bool = False,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """
    Executes a prompt against the active LLM selected in config.py
    and returns a structured JSON string response.

    Supports dynamic mode routing (disable_thinking=True for fast routing,
    disable_thinking=False for deep Q&A / extraction). Each engine formats the
    system/user prompts into its own native chat template.

    This is the public API consumed by extractor.py, rag/qa.py, and rag/router.py.
    """
    engine = _get_engine()
    return engine.generate_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        disable_thinking=disable_thinking,
        max_tokens=max_tokens,
    )


def unload_llm_engine():
    """Unloads the active LLM engine from GPU memory to free VRAM."""
    global _engine_instance
    if _engine_instance is not None:
        logger.info(" Unloading LLM engine to free VRAM...")
        try:
            if hasattr(_engine_instance, "close"):
                _engine_instance.close()
        except Exception as e:
            logger.warning(f" Warning closing LLM engine: {e}")
        _engine_instance = None


__all__ = ["generate_json", "load_llm_engine", "unload_llm_engine"]

