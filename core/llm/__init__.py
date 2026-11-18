"""
core/llm — Pluggable LLM Engine Package & Unified Generation Facade.

Contains concrete LLM engine implementations (e.g. Qwen3_8_27BEngine)
and exposes a clean, model-agnostic `generate_json` and `generate_stream`
interface with per-hop telemetry and prefix cache lifecycle hooks.
"""

import inspect
import logging
import threading
import time
from typing import Any, Dict, List, Optional

from core.registry import get_llm_engine
from server.config import DEFAULT_MAX_TOKENS

logger = logging.getLogger(__name__)

# Singleton LLM instance (cached only after load() succeeds)
_engine_instance = None

#: Serializes ALL LLM inference (chat, extraction, autonomous scrape, etc.).
#: The llama-cpp engine is single-residency (one process, one context) and is NOT
#: thread-safe, so every generate_json / generate_stream call through this facade
#: acquires this lock. RLock allows re-entrancy on the same thread.
_INFERENCE_LOCK = threading.RLock()

#: Thread-local per-turn telemetry collector so each chat turn or API request can
#: inspect the exact sequence of LLM hops and attach them to the terminal event.
_TURN_TELEMETRY = threading.local()


def start_turn_telemetry() -> None:
    """Resets the current thread's per-turn LLM hop telemetry accumulator."""
    _TURN_TELEMETRY.hops = []


def get_turn_telemetry() -> List[Dict[str, Any]]:
    """Returns a copy of all LLM hops recorded on the current thread during this turn."""
    return list(getattr(_TURN_TELEMETRY, "hops", []) or [])


def record_hop_telemetry(hop_data: Dict[str, Any]) -> None:
    """Appends an LLM hop telemetry record to the current thread's turn accumulator."""
    if not hasattr(_TURN_TELEMETRY, "hops") or _TURN_TELEMETRY.hops is None:
        _TURN_TELEMETRY.hops = []
    _TURN_TELEMETRY.hops.append(hop_data)


def _infer_caller_hop_name() -> str:
    """Derives a concise caller function name for telemetry when hop_name is omitted."""
    frame = inspect.currentframe()
    try:
        cur = frame
        while cur is not None:
            fn = cur.f_code.co_name
            mod = cur.f_globals.get("__name__", "")
            if fn not in (
                "_infer_caller_hop_name",
                "generate_json",
                "generate_stream",
                "_generate_raw",
                "_generate_stream",
            ) and not mod.startswith("core.llm"):
                return fn
            cur = cur.f_back
    except Exception:
        pass
    finally:
        del frame
    return "llm_call"


def _get_engine():
    """Get or load the configured LLM engine singleton."""
    global _engine_instance
    if _engine_instance is None:
        from server.config import LLM_ENGINE, CACHE_DIR
        logger.info(f"LLM facade: initializing engine '{LLM_ENGINE}' via registry...")
        candidate = get_llm_engine(LLM_ENGINE)
        candidate.load(CACHE_DIR)
        _engine_instance = candidate
    return _engine_instance


def load_llm_engine():
    """Eagerly load the configured LLM engine into GPU memory on API startup."""
    _get_engine()


def warmup_prefix_cache() -> None:
    """Pre-evaluates and snapshots the base Long-Term Memory prefix into the LLM prefix cache."""
    try:
        engine = _get_engine()
        if hasattr(engine, "warmup_base_prefix"):
            from core.prompts import config_unified_agent_system_prompt
            with _INFERENCE_LOCK:
                engine.warmup_base_prefix(config_unified_agent_system_prompt())
    except Exception as exc:
        logger.warning(f" Could not warm up LLM prefix cache: {exc}")


def invalidate_prefix_cache() -> None:
    """Invalidates cached sequence-state prefix snapshots when Long-Term Memory changes."""
    global _engine_instance
    if _engine_instance is not None and hasattr(_engine_instance, "invalidate_prefix_cache"):
        with _INFERENCE_LOCK:
            _engine_instance.invalidate_prefix_cache()


def _emit_telemetry(engine: Any, hop_label: str, t_wall_ms: float, disable_thinking: bool) -> None:
    """Formats, logs, and records per-hop LLM telemetry."""
    hop_meta = dict(getattr(engine, "last_hop_telemetry", None) or {})
    hop_meta["hop"] = hop_label
    hop_meta.setdefault("total_ms", round(t_wall_ms, 1))
    hop_meta["thinking"] = not disable_thinking

    prompt_tok = hop_meta.get("prompt_tokens", 0)
    cached_tok = hop_meta.get("cached_prefix_tokens", 0)
    eval_tok = hop_meta.get("eval_prompt_tokens", 0)
    gen_tok = hop_meta.get("completion_tokens", 0)
    prefill_ms = hop_meta.get("prefill_ms", 0.0)
    decode_ms = hop_meta.get("decode_ms", 0.0)
    total_ms = hop_meta.get("total_ms", round(t_wall_ms, 1))
    tok_s = hop_meta.get("tok_per_s", 0.0)

    logger.info(
        f"[LLM_TELEMETRY] hop={hop_label} prompt_tok={prompt_tok} cached_tok={cached_tok} "
        f"eval_tok={eval_tok} gen_tok={gen_tok} prefill_ms={prefill_ms} "
        f"decode_ms={decode_ms} total_ms={total_ms} tok_s={tok_s}"
    )
    record_hop_telemetry(hop_meta)


def generate_json(
    system_prompt: str,
    user_prompt: str,
    disable_thinking: bool = False,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    response_prefix: str = "",
    hop_name: str = "",
) -> str:
    """
    Executes a prompt against the active LLM selected in config.py
    and returns a structured JSON string response.
    """
    hop_label = hop_name or _infer_caller_hop_name()
    engine = _get_engine()
    with _INFERENCE_LOCK:
        t0 = time.perf_counter()
        engine.last_hop_telemetry = None
        result = engine.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            disable_thinking=disable_thinking,
            max_tokens=max_tokens,
            response_prefix=response_prefix,
        )
        t_wall_ms = (time.perf_counter() - t0) * 1000.0
        _emit_telemetry(engine, hop_label, t_wall_ms, disable_thinking)
        return result


def generate_stream(
    system_prompt: str,
    user_prompt: str,
    disable_thinking: bool = False,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    response_prefix: str = "",
    hop_name: str = "",
):
    """
    Streams raw model output token-by-token against the active LLM.
    """
    hop_label = hop_name or _infer_caller_hop_name()
    engine = _get_engine()
    with _INFERENCE_LOCK:
        t0 = time.perf_counter()
        engine.last_hop_telemetry = None
        try:
            yield from engine.generate_stream(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                disable_thinking=disable_thinking,
                max_tokens=max_tokens,
                response_prefix=response_prefix,
            )
        finally:
            t_wall_ms = (time.perf_counter() - t0) * 1000.0
            _emit_telemetry(engine, hop_label, t_wall_ms, disable_thinking)


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


__all__ = [
    "generate_json",
    "generate_stream",
    "load_llm_engine",
    "unload_llm_engine",
    "start_turn_telemetry",
    "get_turn_telemetry",
    "record_hop_telemetry",
    "warmup_prefix_cache",
    "invalidate_prefix_cache",
]
