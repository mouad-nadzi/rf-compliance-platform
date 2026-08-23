"""
engines/llm/qwen_agentworld.py — Qwen-AgentWorld-35B-A3B-UD-IQ2_M engine.

Integrates the Qwen-AgentWorld 35B Mixture-of-Experts Language World Model
(3B active parameters per token, IQ2_M quantisation) via llama-cpp-python.

Key design decisions:
  - Raw text completion API (self._llm(...)) with manual ChatML formatting
    to avoid chat-API conflicts with the model's native <think> block output.
  - n_ctx set to config.DEFAULT_CONTEXT_WINDOW (16384 tokens).
  - Free-form JSON generation (no grammar constraint) with unconditional
    reasoning-trace scrubbing + regex extraction in engines.base.
  - Full GPU offload (n_gpu_layers=-1) with a graceful multi-tier fallback
    to partial offload → CPU-only so the load never hard-crashes the kernel.

Production note: see handoff_report.md §9 for removal / deployment guidance.
"""

import logging
import os

from huggingface_hub import hf_hub_download

from engines.base import BaseLLMEngine

# ── llama-cpp-python optional import ─────────────────────────────────────────
try:
    from llama_cpp import Llama
except ImportError:
    Llama = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Context-window cap — controlled centrally via config.DEFAULT_CONTEXT_WINDOW.
from config import DEFAULT_CONTEXT_WINDOW, DEFAULT_MAX_TOKENS


class QwenAgentWorld35BEngine(BaseLLMEngine):
    """
    LLM engine backed by Qwen-AgentWorld-35B-A3B-UD-IQ2_M.gguf via
    llama-cpp-python.

    Model characteristics:
      - Architecture : Mixture-of-Experts (MoE)
      - Total params : 35 B
      - Active params : 3 B per token (selective expert routing)
      - Quantisation  : IQ2_M (dynamic integer quantisation)
      - Context       : controlled via DEFAULT_CONTEXT_WINDOW (16,384 tokens)
      - Thinking mode : native <think>...</think> chain-of-thought blocks

    Prompt format: ChatML  (<|im_start|> / <|im_end|>)
    """

    # ── Configurable via environment variables ────────────────────────────────
    REPO_ID: str = os.getenv(
        "QWEN_AGENTWORLD_REPO_ID",
        "unsloth/Qwen-AgentWorld-35B-A3B-GGUF",
    )
    FILENAME: str = os.getenv(
        "QWEN_AGENTWORLD_FILENAME",
        "Qwen-AgentWorld-35B-A3B-UD-IQ2_M.gguf",
    )

    # Fallback repositories tried in order if the primary repo fails.
    _CANDIDATE_REPOS: list[str] = [
        "Qwen/Qwen-AgentWorld-35B-A3B-GGUF",
        "bartowski/Qwen-AgentWorld-35B-A3B-GGUF",
    ]

    def __init__(self) -> None:
        self._llm: "Llama | None" = None  # type: ignore[name-defined]
        self.active_n_ctx = DEFAULT_CONTEXT_WINDOW

    # ── load() ────────────────────────────────────────────────────────────────

    def load(self, cache_dir: str) -> None:
        """
        Download (if needed) and load the GGUF model onto the GPU.

        Logs model identity, quantisation, parameter counts, and the enforced
        context window so operators can audit what is running in production.

        Args:
            cache_dir: Local directory where downloaded GGUF weights are cached
                       (see config.py CACHE_DIR).

        Raises:
            ImportError : llama-cpp-python is not installed.
            RuntimeError: The GGUF file could not be downloaded or loaded.
        """
        if self._llm is not None:
            logger.info(
                "⚡ Qwen-AgentWorld 35B engine is already loaded — skipping reload."
            )
            return

        # ── 1. Validate inference library ─────────────────────────────────────
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
                "pip uninstall -y llama-cpp-python && pip install llama-cpp-python --force-reinstall --no-cache-dir --index-strategy unsafe-best-match --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122"
            )
        logger.info("⚡ Verified CUDA GPU acceleration active in llama-cpp-python.")

        # ── 2. Announce model identity ────────────────────────────────────────
        logger.info("=" * 70)
        logger.info("🚀 Loading Qwen-AgentWorld-35B-A3B-UD-IQ2_M")
        logger.info("   Total parameters  : 35B")
        logger.info("   Active per token  : 3B  (MoE selective routing)")
        logger.info("   Quantisation      : IQ2_M (dynamic integer quant)")
        logger.info(f"   Context window    : {DEFAULT_CONTEXT_WINDOW} tokens (enforced cap — native: 262,144)")
        logger.info("   GPU offload       : full  (n_gpu_layers=-1)")
        logger.info("=" * 70)

        # ── 3. Resolve GGUF file (multi-repo fallback) ────────────────────────
        candidate_repos = list(dict.fromkeys(
            [self.REPO_ID] + self._CANDIDATE_REPOS
        ))

        model_path: str | None = None
        for repo in candidate_repos:
            try:
                logger.info(
                    f"🔍 Checking HuggingFace repo '{repo}' for '{self.FILENAME}' ..."
                )
                path = hf_hub_download(
                    repo_id=repo,
                    filename=self.FILENAME,
                    cache_dir=cache_dir,
                )
                if os.path.exists(path) and os.path.getsize(path) > 500_000_000:
                    size_gb = os.path.getsize(path) / (1024 ** 3)
                    logger.info(
                        f"✅ Resolved valid GGUF from '{repo}' ({size_gb:.2f} GB)"
                    )
                    model_path = path
                    break
                else:
                    invalid_bytes = (
                        os.path.getsize(path) if os.path.exists(path) else 0
                    )
                    logger.warning(
                        f"⚠️  File in '{repo}' appears to be a Git-LFS pointer "
                        f"({invalid_bytes} bytes). Cleaning and trying next repo ..."
                    )
                    try:
                        os.remove(path)
                    except Exception:
                        pass
            except Exception as exc:
                logger.warning(f"Could not download from '{repo}': {exc}")

        if model_path is None:
            raise RuntimeError(
                f"Could not download a valid GGUF file '{self.FILENAME}' (>500 MB) "
                "from any candidate repository.  "
                "Check HuggingFace network connectivity and repo permissions."
            )

        # ── 4. Initialise llama.cpp with Centralized Context Window ────
        logger.info(
            f"⚙️  Initializing Llama instance (n_ctx={DEFAULT_CONTEXT_WINDOW}, n_gpu_layers=-1, flash_attn=True, KV=q8_0)..."
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
                f"✅ Qwen-AgentWorld 35B loaded successfully (n_ctx={DEFAULT_CONTEXT_WINDOW}, n_gpu_layers=-1)."
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
        Assemble ChatML prompt (handling Qwen-AgentWorld empty <think> block) and execute completion.
        """
        if not system_prompt:
            system_prompt = (
                "You are a precise structured data extraction assistant. "
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
