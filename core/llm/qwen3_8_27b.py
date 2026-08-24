"""
core/llm/qwen3_8_27b.py — Qwen3.8 27B GGUF Model Engine.

Implements Qwen3_8_27BEngine subclassing BaseLLMEngine for local GGUF inference
using pre-compiled llama-cpp-python CUDA acceleration.

Model Weights:
  Repository: unsloth/Qwen3.8-27B-GGUF
  Filename  : Qwen3.8-27B-UD-IQ1_M.gguf (~9.8 GB file size)
"""

import logging
import os
from typing import TYPE_CHECKING, Any, Dict

from server.config import DEFAULT_CONTEXT_WINDOW, DEFAULT_MAX_TOKENS
from core.base import BaseLLMEngine

if TYPE_CHECKING:
    from llama_cpp import Llama

logger = logging.getLogger(__name__)


class Qwen3_8_27BEngine(BaseLLMEngine):
    """
    LLM Engine for Qwen3.8 27B Instruction GGUF (UD-IQ1_M).

    Features:
      - Fits inside ~9.8 GB VRAM budget on Tesla T4 GPUs.
      - Full GPU layer offloading (n_gpu_layers=-1).
      - Context window set via DEFAULT_CONTEXT_WINDOW (16,384 tokens).
      - Hybrid reasoning mode with empty <think> trace suppression for JSON schemas.
    """

    REPO_ID: str = os.getenv(
        "QWEN3_8_27B_REPO_ID", "unsloth/Qwen3.8-27B-GGUF"
    )
    FILENAME: str = os.getenv(
        "QWEN3_8_27B_FILENAME", "Qwen3.8-27B-UD-IQ1_M.gguf"
    )

    _CANDIDATE_REPOS: list[str] = [
        "unsloth/Qwen3.8-27B-GGUF",
        "Qwen/Qwen3.8-27B-GGUF",
        "bartowski/Qwen3.8-27B-GGUF",
    ]

    def __init__(self) -> None:
        self._llm: "Llama | None" = None
        self.active_n_ctx: int = DEFAULT_CONTEXT_WINDOW

    def load(self, cache_dir: str) -> None:
        """Download (if needed) and load Qwen3.8 27B GGUF onto GPU VRAM."""
        if self._llm is not None:
            logger.info("⚡ Qwen3.8 27B engine is already loaded.")
            return

        try:
            from core.utils.vram import flush_gpu_cache
            flush_gpu_cache()
        except Exception as vram_err:
            logger.warning(f"⚠️ VRAM flush warning: {vram_err}")

        try:
            import llama_cpp
            _Llama = llama_cpp.Llama
        except ImportError:
            raise ImportError(
                "llama_cpp is not installed. Please install llama-cpp-python."
            )

        gpu_ok = (
            getattr(llama_cpp, "llama_supports_gpu_offload", lambda: False)() or
            getattr(llama_cpp, "llama_supports_gpu", lambda: False)() or
            getattr(llama_cpp, "llama_supports_cuda", lambda: False)()
        )
        if not gpu_ok:
            raise RuntimeError(
                "❌ CUDA GPU acceleration is NOT active in installed llama-cpp-python! "
                "CPU fallback is strictly disabled to prevent slow inference. "
                "Fix by running:\n"
                "pip uninstall -y llama-cpp-python && pip install --no-cache-dir "
                "https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.34-cu122/llama_cpp_python-0.3.34-py3-none-manylinux_2_35_x86_64.whl"
            )
        logger.info("⚡ Verified CUDA GPU acceleration active in llama-cpp-python.")

        logger.info("=" * 70)
        logger.info(f"🚀 Loading Qwen3.8-27B-UD-IQ1_M (LLM Engine)")
        logger.info(f"   Repository ID     : {self.REPO_ID}")
        logger.info(f"   Filename          : {self.FILENAME}")
        logger.info(f"   Context Window    : {DEFAULT_CONTEXT_WINDOW} tokens")
        logger.info("=" * 70)

        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError("huggingface_hub is not installed. Please install huggingface_hub.")

        candidate_repos = list(dict.fromkeys([self.REPO_ID] + self._CANDIDATE_REPOS))
        model_path = None

        for repo in candidate_repos:
            try:
                logger.info(f"🔍 Checking HuggingFace repo '{repo}' for '{self.FILENAME}'...")
                path = hf_hub_download(
                    repo_id=repo,
                    filename=self.FILENAME,
                    cache_dir=cache_dir,
                )
                if os.path.exists(path) and os.path.getsize(path) > 500_000_000:
                    size_gb = os.path.getsize(path) / (1024 ** 3)
                    logger.info(f"✅ Resolved valid GGUF model from '{repo}' ({size_gb:.2f} GB)")
                    model_path = path
                    break
                else:
                    invalid_bytes = os.path.getsize(path) if os.path.exists(path) else 0
                    logger.warning(
                        f"⚠️ File in '{repo}' is truncated or Git LFS pointer ({invalid_bytes} bytes). Cleaning and trying next repo..."
                    )
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                        except Exception:
                            pass
            except Exception as exc:
                logger.warning(f"Could not download from '{repo}': {exc}")

        if not model_path:
            raise RuntimeError(
                f"Could not download valid GGUF file '{self.FILENAME}' (>500 MB) "
                "from any candidate repository. Please check network connectivity."
            )

        logger.info(
            f"⚙️ Initializing Llama instance (n_ctx={DEFAULT_CONTEXT_WINDOW}, n_gpu_layers=-1, flash_attn=True, KV=q8_0)..."
        )
        try:
            try:
                self._llm = _Llama(
                    model_path=model_path,
                    n_gpu_layers=-1,
                    n_ctx=DEFAULT_CONTEXT_WINDOW,
                    flash_attn=True,
                    type_k=llama_cpp.GGML_TYPE_Q8_0,
                    type_v=llama_cpp.GGML_TYPE_Q8_0,
                    verbose=False,
                )
            except TypeError:
                self._llm = _Llama(
                    model_path=model_path,
                    n_gpu_layers=-1,
                    n_ctx=DEFAULT_CONTEXT_WINDOW,
                    verbose=False,
                )
            self.active_n_ctx = DEFAULT_CONTEXT_WINDOW
            logger.info(
                f"✅ Qwen3.8 27B loaded successfully (n_ctx={DEFAULT_CONTEXT_WINDOW}, n_gpu_layers=-1)."
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load GGUF model from '{model_path}': {exc}")

    # ── BaseLLMEngine hook ───────────────────────────────────────────────────

    def _generate_raw(
        self,
        system_prompt: str,
        user_prompt: str,
        disable_thinking: bool = False,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """
        Assemble ChatML prompt (handling Qwen3.8 empty <think> block) and execute completion.
        """
        if not system_prompt:
            system_prompt = (
                "You are an expert technical data extraction engine. "
                "Respond with a valid JSON object only."
            )

        prompt = (
            f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        if disable_thinking and "<think>" not in prompt:
            prompt = prompt.replace(
                "<|im_start|>assistant\n",
                "<|im_start|>assistant\n<think>\n</think>\n",
                1,
            )

        temp = 0.0 if disable_thinking else 0.6
        top_p = 0.80 if disable_thinking else 0.95

        response = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temp,
            top_p=top_p,
            top_k=20,
            min_p=0.0,
            presence_penalty=1.5,
            repeat_penalty=1.0,
            echo=False,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        return response["choices"][0]["text"]


# Alias for compatibility with direct imports
Qwen3827BEngine = Qwen3_8_27BEngine
