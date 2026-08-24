"""
core/llm/gemma4_26b.py — Gemma 4 26B-A4B GGUF Model Engine.

Implements Gemma4_26BEngine subclassing BaseLLMEngine for local GGUF inference
using pre-compiled llama-cpp-python CUDA acceleration.

Model Weights:
  Repository: unsloth/gemma-4-26B-A4B-it-GGUF
  Filename  : gemma-4-26B-A4B-it-UD-IQ2_M.gguf (~10.2 GB file size)
"""

import logging
import os
from typing import TYPE_CHECKING, Any, Dict

from server.config import DEFAULT_CONTEXT_WINDOW, DEFAULT_MAX_TOKENS
from core.base import BaseLLMEngine

if TYPE_CHECKING:
    from llama_cpp import Llama

logger = logging.getLogger(__name__)


class Gemma4_26BEngine(BaseLLMEngine):
    """
    LLM Engine for Gemma 4 26B-A4B Instruction GGUF (UD-IQ2_M).

    Features:
      - Fits inside ~10.2 GB VRAM budget on Tesla T4 GPUs.
      - Full GPU layer offloading (n_gpu_layers=-1).
      - Context window set via DEFAULT_CONTEXT_WINDOW (16,384 tokens).
      - Structured JSON generation.
    """

    REPO_ID: str = os.getenv(
        "GEMMA4_26B_REPO_ID", "unsloth/gemma-4-26B-A4B-it-GGUF"
    )
    FILENAME: str = os.getenv(
        "GEMMA4_26B_FILENAME", "gemma-4-26B-A4B-it-UD-IQ2_M.gguf"
    )

    _CANDIDATE_REPOS: list[str] = [
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "google/gemma-4-26B-A4B-it-GGUF",
    ]

    def __init__(self) -> None:
        self._llm: "Llama | None" = None
        self.active_n_ctx: int = DEFAULT_CONTEXT_WINDOW

    def load(self, cache_dir: str) -> None:
        """Download (if needed) and load Gemma 4 26B GGUF onto GPU VRAM."""
        if self._llm is not None:
            logger.info(" Gemma 4 26B engine is already loaded.")
            return

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
                " CUDA GPU acceleration is NOT active in installed llama-cpp-python! "
                "CPU fallback is strictly disabled to prevent slow inference. "
                "Fix by running:\n"
                "pip uninstall -y llama-cpp-python && pip install --no-cache-dir "
                "https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.19-cu122/llama_cpp_python-0.3.19-cp312-cp312-linux_x86_64.whl"
            )
        logger.info(" Verified CUDA GPU acceleration active in llama-cpp-python.")

        logger.info("=" * 70)
        logger.info(f" Loading Gemma 4 26B-A4B-it (UD-IQ2_M GGUF Engine)")
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
                logger.info(f" Checking HuggingFace repo '{repo}' for '{self.FILENAME}'...")
                path = hf_hub_download(
                    repo_id=repo,
                    filename=self.FILENAME,
                    cache_dir=cache_dir,
                )
                if os.path.exists(path) and os.path.getsize(path) > 500_000_000:
                    size_gb = os.path.getsize(path) / (1024 ** 3)
                    logger.info(f" Resolved valid GGUF model from '{repo}' ({size_gb:.2f} GB)")
                    model_path = path
                    break
                else:
                    invalid_bytes = os.path.getsize(path) if os.path.exists(path) else 0
                    logger.warning(
                        f" File in '{repo}' is truncated or Git LFS pointer ({invalid_bytes} bytes). Cleaning and trying next repo..."
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
            f" Initializing Llama instance (n_ctx={DEFAULT_CONTEXT_WINDOW}, n_gpu_layers=-1, flash_attn=True, KV=q8_0)..."
        )
        try:
            try:
                # NOTE: type_k/type_v must be the integer GGML enum, NOT the string
                # "q8_0". Strings raise TypeError on llama-cpp-python 0.3.34 and
                # silently fall back to a no-flash-attn path that pads the V cache
                # to 2048 — blowing the VRAM budget and failing llama_context creation.
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
                f" Gemma 4 26B loaded successfully (n_ctx={DEFAULT_CONTEXT_WINDOW}, n_gpu_layers=-1)."
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
        Assemble the Gemma 4 native turn-format prompt and execute completion.
        """
        blocks = []
        if system_prompt:
            blocks.append(f"<start_of_turn>system\n{system_prompt}<end_of_turn>\n")
        blocks.append(f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n")
        blocks.append("<start_of_turn>model\n")
        prompt = "".join(blocks)

        response = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.8,
            echo=False,
            stop=["<end_of_turn>", "<|im_end|>", "<|endoftext|>"],
        )
        return response["choices"][0]["text"]
