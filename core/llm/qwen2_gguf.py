"""
core/llm/qwen2_gguf.py — Qwen2-7B-Instruct GGUF engine implementation.

Runs the structured data extraction completely locally using the GPU
via llama-cpp-python, without relying on any external APIs.

Model: Qwen/Qwen2-7B-Instruct-GGUF (Q4_K_M quantization)
"""

import logging
import os
import shutil

# pyrefly: ignore [missing-import]
from huggingface_hub import hf_hub_download

from core.base import BaseLLMEngine

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

logger = logging.getLogger(__name__)

# Context-window cap — controlled centrally via config.
from server.config import DEFAULT_CONTEXT_WINDOW, DEFAULT_MAX_TOKENS


class Qwen2GGUFEngine(BaseLLMEngine):
    """
    LLM engine backed by Qwen2-7B-Instruct-Q4_K_M.gguf via llama-cpp-python.

    Fully offloaded to GPU (n_gpu_layers=-1). Context window is controlled by
    DEFAULT_CONTEXT_WINDOW (16,384 tokens) in config.py.
    """

    REPO_ID = "Qwen/Qwen2-7B-Instruct-GGUF"
    FILENAME = "qwen2-7b-instruct-q4_k_m.gguf"

    def __init__(self):
        self._llm = None
        self.active_n_ctx = DEFAULT_CONTEXT_WINDOW

    # ── BaseLLMEngine interface ───────────────────────────────────────────

    def load(self, cache_dir: str) -> None:
        """Download (if needed) and load the GGUF model onto the GPU."""
        if self._llm is not None:
            logger.info(" Qwen GGUF model is already loaded, skipping reload.")
            return

        try:
            import llama_cpp
            Llama = llama_cpp.Llama
        except ImportError:
            raise ImportError("llama_cpp is not installed. Please install llama-cpp-python.")

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
                "pip uninstall -y llama-cpp-python && pip install llama-cpp-python --force-reinstall --no-cache-dir --index-strategy unsafe-best-match --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122"
            )
        logger.info(" Verified CUDA GPU acceleration active in llama-cpp-python.")

        logger.info(f"Checking for local GGUF model: {self.FILENAME}")
        model_path = hf_hub_download(
            repo_id=self.REPO_ID,
            filename=self.FILENAME,
            cache_dir=cache_dir,
        )
        logger.info(f"Model path resolved to: {model_path}")

        # Check if downloaded file is corrupted or a tiny Git LFS pointer file (<100MB for GGUF model)
        if os.path.exists(model_path) and os.path.getsize(model_path) < 100_000_000:
            logger.error(
                f" Model file at '{model_path}' is invalid/corrupted "
                f"({os.path.getsize(model_path)} bytes). Deleting invalid file to force fresh download..."
            )
            try:
                os.remove(model_path)
            except Exception:
                pass

            logger.info(f" Re-downloading {self.FILENAME} from Hugging Face hub...")
            model_path = hf_hub_download(
                repo_id=self.REPO_ID,
                filename=self.FILENAME,
                cache_dir=cache_dir,
                force_download=True,
            )

        logger.info("Initializing Qwen GGUF Model with llama.cpp...")
        try:
            try:
                self._llm = Llama(
                    model_path=model_path,
                    n_gpu_layers=-1,
                    n_ctx=DEFAULT_CONTEXT_WINDOW,
                    flash_attn=True,
                    type_k="q8_0",
                    type_v="q8_0",
                    verbose=False,
                )
            except TypeError:
                self._llm = Llama(
                    model_path=model_path,
                    n_gpu_layers=-1,
                    n_ctx=DEFAULT_CONTEXT_WINDOW,
                    verbose=False,
                )
            self.active_n_ctx = DEFAULT_CONTEXT_WINDOW
            logger.info(f" Qwen2 loaded successfully (n_ctx={DEFAULT_CONTEXT_WINDOW}, n_gpu_layers=-1).")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load GGUF model from file '{model_path}': {exc}. "
                "Please verify that the GGUF file is fully downloaded and sufficient VRAM/RAM is free."
            )

    # ── BaseLLMEngine hook ───────────────────────────────────────────────────

    def _generate_raw(
        self,
        system_prompt: str,
        user_prompt: str,
        disable_thinking: bool = False,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """Assemble ChatML prompt and execute completion with json_object constraint."""
        if not system_prompt:
            system_prompt = "You are a helpful assistant designed to output strict JSON."
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        temp = 0.1 if disable_thinking else 0.7

        try:
            response = self._llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temp,
                response_format={"type": "json_object"},
                echo=False,
            )
        except TypeError:
            response = self._llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temp,
                echo=False,
            )
        return response["choices"][0]["text"]
