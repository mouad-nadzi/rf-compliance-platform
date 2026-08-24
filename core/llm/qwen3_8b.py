"""
core/llm/qwen3_8b.py — Unified Qwen3-8B-Q8_0 GGUF Engine Implementation.

Standardizes the pipeline on a single, unified LLM (Qwen3-8B-Q8_0.gguf)
capable of dynamic execution:
  1. Fast Non-Thinking Mode (disable_thinking=True): Appends /no_think,
     temp=0.7, top_p=0.8, free-form output with trace scrubbing for sub-second intent classification.
  2. Deep Thinking Mode (disable_thinking=False): Appends /think,
     temp=0.6, top_p=0.95, unconstrained CoT reasoning for Q&A synthesis and extraction.

Model: Qwen/Qwen3-8B-GGUF (Qwen3-8B-Q8_0.gguf)
"""

import logging
import os

from huggingface_hub import hf_hub_download
from core.base import BaseLLMEngine
from server.config import DEFAULT_CONTEXT_WINDOW, DEFAULT_MAX_TOKENS

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class Qwen3_8BGGUFEngine(BaseLLMEngine):
    """
    LLM engine backed by Qwen3-8B-Q8_0.gguf via llama-cpp-python.

    Fully offloaded to GPU (n_gpu_layers=-1). FlashAttention and 8-bit KV caching
    (type_k="q8_0", type_v="q8_0") enable a 16,384-token context window
    alongside 8.7 GB model weights inside 16 GB VRAM.
    """

    REPO_ID: str = os.getenv("QWEN3_8B_REPO_ID", "Qwen/Qwen3-8B-GGUF")
    FILENAME: str = os.getenv("QWEN3_8B_FILENAME", "Qwen3-8B-Q8_0.gguf")

    _CANDIDATE_REPOS: list[str] = [
        "Qwen/Qwen3-8B-GGUF",
        "unsloth/Qwen3-8B-GGUF",
        "bartowski/Qwen3-8B-GGUF",
    ]

    def __init__(self) -> None:
        self._llm: "Llama | None" = None  # type: ignore[name-defined]
        self.active_n_ctx: int = DEFAULT_CONTEXT_WINDOW

    def load(self, cache_dir: str) -> None:
        """Download (if needed) and load the Qwen3-8B GGUF model onto GPU."""
        if self._llm is not None:
            logger.info("⚡ Qwen3 8B engine is already loaded.")
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
                "❌ CUDA GPU acceleration is NOT active in installed llama-cpp-python! "
                "CPU fallback is strictly disabled to prevent slow inference. "
                "Fix by running:\n"
                "pip uninstall -y llama-cpp-python && pip install llama-cpp-python==0.3.19 --force-reinstall --no-cache-dir --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122"
            )
        logger.info("⚡ Verified CUDA GPU acceleration active in llama-cpp-python.")

        logger.info("=" * 70)
        logger.info(f"🚀 Loading Qwen3-8B-Q8_0 (Unified LLM Engine)")
        logger.info(f"   Repository ID     : {self.REPO_ID}")
        logger.info(f"   Filename          : {self.FILENAME}")
        logger.info(f"   Context Window    : {DEFAULT_CONTEXT_WINDOW} tokens")
        logger.info("   KV Cache Quant    : q8_0 Key/Value + FlashAttention")
        logger.info("=" * 70)

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
                    type_k="q8_0",
                    type_v="q8_0",
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
                f"✅ Qwen3 8B loaded successfully (n_ctx={DEFAULT_CONTEXT_WINDOW}, n_gpu_layers=-1)."
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
        Assemble ChatML prompt with Qwen3 soft mode tag and execute completion.
        """
        if not system_prompt:
            system_prompt = (
                "You are a precise structured data extraction assistant. "
                "Respond with a valid JSON object only."
            )
        mode_tag = " /no_think" if disable_thinking else " /think"
        if (
            mode_tag not in user_prompt
            and "/no_think" not in user_prompt
            and "/think" not in user_prompt
        ):
            user_prompt = f"{user_prompt}{mode_tag}"

        prompt = (
            f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}\n<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        temp = 0.7 if disable_thinking else 0.6
        top_p = 0.8 if disable_thinking else 0.95

        response = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temp,
            top_p=top_p,
            echo=False,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        return response["choices"][0]["text"]
