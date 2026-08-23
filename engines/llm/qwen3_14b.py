"""
engines/llm/qwen3_14b.py — Qwen3-14B-GGUF engine implementation.

Integrates Qwen3-14B in "Pure Instruct" mode (disabling thinking)
with strict sampling parameters as per official recommendations.

Model: Qwen/Qwen3-14B-GGUF (Qwen3-14B-Q4_K_M.gguf)
"""

import logging
import os

# pyrefly: ignore [missing-import]
from huggingface_hub import hf_hub_download

from engines.base import BaseLLMEngine

try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import register_chat_format, Jinja2ChatFormatter
except ImportError:
    Llama = None
    register_chat_format = None
    Jinja2ChatFormatter = None

logger = logging.getLogger(__name__)

# Context-window cap — controlled centrally via config.
from config import DEFAULT_CONTEXT_WINDOW, DEFAULT_MAX_TOKENS

# Register custom chat format for Qwen3 Pure Instruct mode
_QWEN3_PURE_INSTRUCT_TEMPLATE = (
    "{%- set enable_thinking = false %}\n"
    "{%- for message in messages %}"
    "{%- if message['role'] == 'system' %}"
    "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
    "{%- elif message['role'] == 'user' %}"
    "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
    "{%- elif message['role'] == 'assistant' %}"
    "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

if register_chat_format and Jinja2ChatFormatter:
    try:
        _qwen3_formatter_instance = Jinja2ChatFormatter(
            template=_QWEN3_PURE_INSTRUCT_TEMPLATE,
            eos_token="<|im_end|>",
            bos_token="<|im_start|>"
        )

        @register_chat_format("qwen3_pure_instruct")
        def _qwen3_pure_instruct_formatter(messages, **kwargs):
            return _qwen3_formatter_instance(messages=messages, **kwargs)
    except Exception as e:
        logger.warning(f"Failed to register qwen3_pure_instruct chat format: {e}")


class Qwen3_14B_GGUFEngine(BaseLLMEngine):
    """
    LLM engine backed by Qwen3-14B-Q4_K_M.gguf via llama-cpp-python.

    Fully offloaded to GPU (n_gpu_layers=-1). Configured for "Pure Instruct"
    mode with no `<think>` tags and strict sampling parameters. Context window
    is controlled by DEFAULT_CONTEXT_WINDOW (16,384 tokens) in config.py.
    """

    REPO_ID = os.getenv("QWEN3_REPO_ID", "unsloth/Qwen3-14B-GGUF")
    FILENAME = os.getenv("QWEN3_FILENAME", "Qwen3-14B-UD-IQ1_M.gguf")

    def __init__(self):
        self._llm = None
        self.active_n_ctx = DEFAULT_CONTEXT_WINDOW

    def load(self, cache_dir: str) -> None:
        """Download (if needed) and load the GGUF model onto the GPU."""
        if self._llm is not None:
            logger.info("⚡ Qwen3 14B GGUF model is already loaded, skipping reload.")
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
                "❌ CUDA GPU acceleration is NOT active in installed llama-cpp-python! "
                "CPU fallback is strictly disabled to prevent slow inference. "
                "Fix by running:\n"
                "pip uninstall -y llama-cpp-python && pip install llama-cpp-python --force-reinstall --no-cache-dir --index-strategy unsafe-best-match --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122"
            )
        logger.info("⚡ Verified CUDA GPU acceleration active in llama-cpp-python.")

        # Multi-repository resolution: find a valid >500MB GGUF model file
        candidate_repos = list(dict.fromkeys([
            self.REPO_ID,
            "unsloth/Qwen3-14B-GGUF",
            "bartowski/Qwen_Qwen3-14B-GGUF",
            "6block/Qwen3-14B-GGUF",
        ]))

        model_path = None
        for repo in candidate_repos:
            try:
                logger.info(f"Checking Hugging Face repo '{repo}' for '{self.FILENAME}'...")
                path = hf_hub_download(
                    repo_id=repo,
                    filename=self.FILENAME,
                    cache_dir=cache_dir,
                )
                if os.path.exists(path) and os.path.getsize(path) > 500_000_000:
                    model_path = path
                    size_gb = os.path.getsize(path) / (1024 ** 3)
                    logger.info(f"✅ Resolved valid GGUF model from '{repo}' ({size_gb:.2f} GB)")
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
            except Exception as e:
                logger.warning(f"Could not download from '{repo}': {e}")

        if not model_path:
            raise RuntimeError(
                f"Could not download valid GGUF file '{self.FILENAME}' (>500MB) from any candidate repository. "
                "Please verify Hugging Face network connectivity."
            )

        logger.info(f"Attempting Llama initialization (n_ctx={DEFAULT_CONTEXT_WINDOW}, n_gpu_layers=-1, flash_attn=True, KV=q8_0)...")
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
            logger.info(f"✅ Qwen3 14B loaded successfully (n_ctx={DEFAULT_CONTEXT_WINDOW}, n_gpu_layers=-1).")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load GGUF model from '{model_path}': {exc}. "
                "Please verify that the GGUF file is fully downloaded (~9GB) and sufficient VRAM/RAM is free."
            )

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
