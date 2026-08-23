"""
engines/registry.py — Factory/Registry for pluggable OCR and LLM engines.

Maps human-readable string keys (e.g., "glm-ocr", "got-ocr2") to concrete engine classes.
Uses lazy imports so only the selected model's dependencies are loaded.

Usage:
    from engines.registry import get_ocr_engine, get_llm_engine

    ocr = get_ocr_engine("glm-ocr")
    ocr.load(cache_dir="/path/to/cache")
    markdown = ocr.process_document("doc.pdf", "/tmp/out")
"""

from engines.base import BaseOCREngine, BaseLLMEngine


# ── Registry Dictionaries ────────────────────────────────────────────────────
# Each entry maps:  "key" -> "module_path.ClassName"
# Only the selected model is imported at runtime (lazy loading).

OCR_REGISTRY: dict[str, str] = {
    "deepseek-ocr-2": "engines.ocr.deepseek_ocr2.DeepSeekOCR2Engine",
    "glm-ocr":        "engines.ocr.glm_ocr.GLMOCREngine",
    "got-ocr2":       "engines.ocr.got_ocr2.GOTOCR2Engine",
}

LLM_REGISTRY: dict[str, str] = {
    "qwen3.6-35b-gguf":       "engines.llm.qwen3_35b.Qwen3_35BEngine",
    "qwen3-35b":              "engines.llm.qwen3_35b.Qwen3_35BEngine",
    "qwen3.8-27b-gguf":       "engines.llm.qwen3_8_27b.Qwen3_8_27BEngine",
    "qwen3.8-27b":            "engines.llm.qwen3_8_27b.Qwen3_8_27BEngine",
    "gemma4-26b-gguf":        "engines.llm.gemma4_26b.Gemma4_26BEngine",
    "qwen3-8b":               "engines.llm.qwen3_8b.Qwen3_8BGGUFEngine",
    "qwen2-7b-gguf":          "engines.llm.qwen2_gguf.Qwen2GGUFEngine",
    "qwen3-14b-gguf":         "engines.llm.qwen3_14b.Qwen3_14B_GGUFEngine",
    "qwen-agentworld-35b":    "engines.llm.qwen_agentworld.QwenAgentWorld35BEngine",
}


# ── Factory Functions ─────────────────────────────────────────────────────────

def _import_class(dotted_path: str):
    """
    Dynamically import a class from a dotted module path.
    Example: "engines.ocr.glm_ocr.GLMOCREngine" → <class GLMOCREngine>
    """
    module_path, class_name = dotted_path.rsplit(".", 1)

    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_ocr_engine(key: str) -> BaseOCREngine:
    """
    Instantiate and return the OCR engine registered under `key`.

    Args:
        key: A string matching a key in OCR_REGISTRY (e.g., "glm-ocr").

    Returns:
        An instance of the corresponding BaseOCREngine subclass.

    Raises:
        ValueError: If the key is not found in the registry.
    """
    if key not in OCR_REGISTRY:
        available = ", ".join(sorted(OCR_REGISTRY.keys()))
        raise ValueError(
            f"Unknown OCR engine '{key}'. Available engines: {available}"
        )

    cls = _import_class(OCR_REGISTRY[key])
    return cls()


def get_llm_engine(key: str) -> BaseLLMEngine:
    """
    Instantiate and return the LLM engine registered under `key`.

    Args:
        key: A string matching a key in LLM_REGISTRY (e.g., "qwen2-7b-gguf").

    Returns:
        An instance of the corresponding BaseLLMEngine subclass.

    Raises:
        ValueError: If the key is not found in the registry.
    """
    if key not in LLM_REGISTRY:
        available = ", ".join(sorted(LLM_REGISTRY.keys()))
        raise ValueError(
            f"Unknown LLM engine '{key}'. Available engines: {available}"
        )

    cls = _import_class(LLM_REGISTRY[key])
    return cls()
