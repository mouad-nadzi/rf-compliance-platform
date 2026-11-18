"""
core/llm/qwen3_8_27b.py — Qwen3.8 27B GGUF Model Engine.

Implements Qwen3_8_27BEngine subclassing BaseLLMEngine for local GGUF inference
using pre-compiled llama-cpp-python CUDA acceleration, sequence-state prefix
checkpointing for hybrid recurrent (qwen35) architectures, and native llama.cpp
hardware telemetry counters.

Model Weights:
  Repository: unsloth/Qwen3.8-27B-GGUF
  Filename  : Qwen3.8-27B-UD-IQ3_XXS.gguf (~10.93 GB file size, 3-bit)
"""

from collections import OrderedDict
import ctypes
import logging
import os
import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from server.config import DEFAULT_CONTEXT_WINDOW, DEFAULT_MAX_TOKENS
from core.base import BaseLLMEngine

if TYPE_CHECKING:
    from llama_cpp import Llama

logger = logging.getLogger(__name__)

_MEMORY_END_MARKER = "[END LONG-TERM MEMORY]"
_HISTORY_START_MARKER = "--- CONVERSATION HISTORY ---\n"
_HISTORY_END_MARKER = "\n--- END CONVERSATION HISTORY ---"
_TURN_SPLIT_RE = re.compile(r"\n(?=(?:User|Assistant): )")


class Qwen3_8_27BEngine(BaseLLMEngine):
    """
    LLM Engine for Qwen3.8 27B Instruction GGUF (UD-IQ3_XXS).

    Features:
      - Co-resident in VRAM (~11 GB) alongside GLM-OCR on the GCP NVIDIA L4 (24 GB).
      - Full GPU layer offloading (n_gpu_layers=-1) with n_batch=2048, n_ubatch=2048.
      - Native Qwen3.8 ChatML thinking switch (`<think>\\n\\n</think>\\n\\n` when
        disable_thinking=True, matching the GGUF tokenizer.chat_template).
      - Sequence-state prefix checkpoint cache (`llama_state_seq_get_data` /
        `llama_state_seq_set_data`) enabling 100% prefix reuse of Long-Term Memory
        and incremental conversation history on hybrid recurrent (`qwen35`) layers.
    """

    REPO_ID: str = os.getenv(
        "QWEN3_8_27B_REPO_ID", "unsloth/Qwen3.8-27B-GGUF"
    )
    FILENAME: str = os.getenv(
        "QWEN3_8_27B_FILENAME", "Qwen3.8-27B-UD-IQ3_XXS.gguf"
    )
    N_GPU_LAYERS: int = int(os.getenv("QWEN3_8_27B_N_GPU_LAYERS", "-1"))
    N_BATCH: int = int(os.getenv("QWEN3_8_27B_N_BATCH", "2048"))
    MAX_PREFIX_CACHE_BYTES: int = int(os.getenv("LLM_PREFIX_CACHE_MB", "1536")) * 1024 * 1024

    _CANDIDATE_REPOS: list[str] = [
        "unsloth/Qwen3.8-27B-GGUF",
        "Qwen/Qwen3.8-27B-GGUF",
        "bartowski/Qwen3.8-27B-GGUF",
    ]

    def __init__(self) -> None:
        self._llm: "Llama | None" = None
        self.active_n_ctx: int = DEFAULT_CONTEXT_WINDOW
        self.last_hop_telemetry: Optional[Dict[str, Any]] = None
        self._prefix_state_cache: "OrderedDict[Tuple[int, ...], bytes]" = OrderedDict()
        self._pinned_base_key: Optional[Tuple[int, ...]] = None
        self._seg_token_cache: Dict[Tuple[str, bool], Tuple[int, ...]] = {}

    def load(self, cache_dir: str) -> None:
        """Download (if needed) and load Qwen3.8 27B GGUF onto GPU VRAM."""
        if self._llm is not None:
            logger.info(" Qwen3.8 27B engine is already loaded.")
            return

        try:
            from core.utils.vram import flush_gpu_cache
            flush_gpu_cache()
        except Exception as vram_err:
            logger.warning(f" VRAM flush warning: {vram_err}")

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
                "https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.34-cu122/llama_cpp_python-0.3.34-py3-none-manylinux_2_35_x86_64.whl"
            )
        logger.info(" Verified CUDA GPU acceleration active in llama-cpp-python.")

        logger.info("=" * 70)
        logger.info(" Loading Qwen3.8-27B-UD-IQ3_XXS (LLM Engine)")
        logger.info(f"   Repository ID     : {self.REPO_ID}")
        logger.info(f"   Filename          : {self.FILENAME}")
        logger.info(f"   Context Window    : {DEFAULT_CONTEXT_WINDOW} tokens")
        logger.info(f"   Batch Size        : {self.N_BATCH} tokens")
        logger.info(f"   GPU Layers        : {self.N_GPU_LAYERS} (-1 = all)")
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
            f" Initializing Llama instance (n_ctx={DEFAULT_CONTEXT_WINDOW}, n_batch={self.N_BATCH}, n_gpu_layers={self.N_GPU_LAYERS}, flash_attn=True, KV=q8_0)..."
        )
        try:
            try:
                self._llm = _Llama(
                    model_path=model_path,
                    n_gpu_layers=self.N_GPU_LAYERS,
                    n_ctx=DEFAULT_CONTEXT_WINDOW,
                    n_batch=self.N_BATCH,
                    n_ubatch=self.N_BATCH,
                    flash_attn=True,
                    type_k=llama_cpp.GGML_TYPE_Q8_0,
                    type_v=llama_cpp.GGML_TYPE_Q8_0,
                    verbose=False,
                )
            except TypeError:
                self._llm = _Llama(
                    model_path=model_path,
                    n_gpu_layers=self.N_GPU_LAYERS,
                    n_ctx=DEFAULT_CONTEXT_WINDOW,
                    n_batch=self.N_BATCH,
                    verbose=False,
                )
            self.active_n_ctx = DEFAULT_CONTEXT_WINDOW
            logger.info(
                f" Qwen3.8 27B loaded successfully (n_ctx={DEFAULT_CONTEXT_WINDOW}, n_batch={self.N_BATCH}, n_gpu_layers={self.N_GPU_LAYERS})."
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load GGUF model from '{model_path}': {exc}")

    # ── Prefix Checkpoint Cache Management ───────────────────────────────────

    def invalidate_prefix_cache(self) -> None:
        """Clears all cached sequence-state checkpoints when Long-Term Memory is modified."""
        self._prefix_state_cache.clear()
        self._pinned_base_key = None
        self._seg_token_cache.clear()
        logger.info(" Cleared LLM sequence-state prefix checkpoint cache.")

    def warmup_base_prefix(self, sample_system_prompt: str) -> None:
        """Pre-evaluates and pins the base Long-Term Memory prefix checkpoint at startup."""
        if self._llm is None:
            return
        prefix_segs, _ = self._split_prompt_segments(
            system_prompt=sample_system_prompt,
            user_prompt="warmup",
            disable_thinking=True,
            response_prefix="",
        )
        if not prefix_segs:
            return
        t0 = time.perf_counter()
        self._prepare_cached_prompt_tokens(prefix_segs, "warmup")
        ms = (time.perf_counter() - t0) * 1000.0
        if self._pinned_base_key:
            size_mb = len(self._prefix_state_cache.get(self._pinned_base_key, b"")) / (1024 * 1024)
            logger.info(
                f" Warmed up base Long-Term Memory prefix cache "
                f"({len(self._pinned_base_key)} tokens, {size_mb:.1f} MB state, {ms:.0f} ms)."
            )

    def _tokenize_segment(self, text: str, is_first: bool = False) -> Tuple[int, ...]:
        """Tokenizes a prompt segment with memoization."""
        key = (text, is_first)
        cached = self._seg_token_cache.get(key)
        if cached is not None:
            return cached

        raw_tokens = self._llm.tokenize(text.encode("utf-8"), add_bos=False, special=True)
        if is_first and self._llm._model.add_bos_token():
            cls_id = self._llm._model.token_cls()
            bos_id = cls_id if cls_id != -1 else self._llm.token_bos()
            if bos_id != -1:
                raw_tokens = [bos_id] + raw_tokens

        tok_tuple = tuple(raw_tokens)
        if len(self._seg_token_cache) > 512:
            self._seg_token_cache.clear()
        self._seg_token_cache[key] = tok_tuple
        return tok_tuple

    def _save_seq0_state(self) -> Optional[bytes]:
        """Serializes active sequence 0 KV + recurrent state from llama.cpp into bytes."""
        import llama_cpp
        try:
            ctx_ptr = self._llm._ctx.ctx
            state_size = int(llama_cpp.llama_state_seq_get_size(ctx_ptr, 0))
            if state_size <= 0:
                return None
            buf = (ctypes.c_uint8 * state_size)()
            n_written = int(llama_cpp.llama_state_seq_get_data(ctx_ptr, buf, state_size, 0))
            if n_written <= 0 or n_written > state_size:
                return None
            return ctypes.string_at(buf, n_written)
        except Exception as exc:
            logger.warning(f" Could not snapshot sequence state: {exc}")
            return None

    def _restore_seq0_state(self, prefix_tokens: Tuple[int, ...], state_bytes: bytes) -> bool:
        """Restores serialized sequence 0 KV + recurrent state back into llama.cpp."""
        import llama_cpp
        try:
            ctx_ptr = self._llm._ctx.ctx
            self._llm.reset()
            size = len(state_bytes)
            buf = (ctypes.c_uint8 * size).from_buffer_copy(state_bytes)
            n_read = int(llama_cpp.llama_state_seq_set_data(ctx_ptr, buf, size, 0))
            if n_read <= 0:
                return False
            n_tok = len(prefix_tokens)
            self._llm.input_ids[:n_tok] = prefix_tokens
            self._llm.input_ids[n_tok:] = -1
            self._llm.n_tokens = n_tok
            self._llm._requires_eval = True
            return True
        except Exception as exc:
            logger.warning(f" Could not restore sequence state: {exc}")
            self._llm.reset()
            return False

    def _store_prefix_checkpoint(self, prefix_key: Tuple[int, ...], state_bytes: Optional[bytes], pin: bool = False) -> None:
        """Stores a sequence state snapshot in the bounded LRU cache."""
        if not state_bytes:
            return
        if pin and self._pinned_base_key is None:
            self._pinned_base_key = prefix_key
        if prefix_key in self._prefix_state_cache:
            del self._prefix_state_cache[prefix_key]
        self._prefix_state_cache[prefix_key] = state_bytes

        total_bytes = sum(len(v) for v in self._prefix_state_cache.values())
        while total_bytes > self.MAX_PREFIX_CACHE_BYTES and len(self._prefix_state_cache) > 1:
            oldest_key = next(iter(self._prefix_state_cache))
            if oldest_key == self._pinned_base_key and len(self._prefix_state_cache) > 1:
                # Keep pinned base key; evict the next oldest entry
                keys_iter = iter(self._prefix_state_cache)
                next(keys_iter)
                second_key = next(keys_iter)
                removed = self._prefix_state_cache.pop(second_key)
                total_bytes -= len(removed)
            else:
                removed = self._prefix_state_cache.pop(oldest_key)
                total_bytes -= len(removed)

    def _assistant_header(self, disable_thinking: bool, response_prefix: str = "") -> str:
        """
        Builds the official Qwen3.8 ChatML assistant generation prompt.
        Per the Qwen3.8 GGUF tokenizer.chat_template:
          - enable_thinking=false (disable_thinking=True): `<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n`
          - enable_thinking=true  (disable_thinking=False): `<|im_start|>assistant\\n<think>\\n` (unless response_prefix overrides)
        """
        if disable_thinking:
            return f"<|im_start|>assistant\n<think>\n\n</think>\n\n{response_prefix}"
        if response_prefix:
            return f"<|im_start|>assistant\n{response_prefix}"
        return "<|im_start|>assistant\n<think>\n"

    def _clean_user_prompt(self, user_prompt: str) -> str:
        """Strips legacy /no_think or /think tags if callers appended them."""
        cleaned = user_prompt
        if cleaned.endswith(" /no_think"):
            cleaned = cleaned[:-10]
        elif cleaned.endswith(" /think"):
            cleaned = cleaned[:-7]
        return cleaned

    def _split_prompt_segments(
        self,
        system_prompt: str,
        user_prompt: str,
        disable_thinking: bool,
        response_prefix: str = "",
    ) -> Tuple[List[str], str]:
        """
        Decomposes (system_prompt, user_prompt) into:
          1. prefix_segments: cacheable prefix segments [base_memory_segment, history_turn_1, ..., history_turn_k]
          2. suffix_str: task-specific instructions + current user payload + assistant header
        """
        if not system_prompt:
            system_prompt = (
                "You are an expert technical data extraction engine. "
                "Respond with a valid JSON object only."
            )
        clean_user = self._clean_user_prompt(user_prompt)
        assistant_hdr = self._assistant_header(disable_thinking, response_prefix)

        mem_idx = system_prompt.find(_MEMORY_END_MARKER)
        if mem_idx == -1:
            full_str = (
                f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
                f"<|im_start|>user\n{clean_user}\n<|im_end|>\n"
                f"{assistant_hdr}"
            )
            return [], full_str

        mem_end = mem_idx + len(_MEMORY_END_MARKER)
        mem_block = system_prompt[:mem_end].strip()
        task_sys = system_prompt[mem_end:].strip()

        prefix_segments: List[str] = [f"<|im_start|>system\n{mem_block}\n"]

        if clean_user.startswith(_HISTORY_START_MARKER) and _HISTORY_END_MARKER in clean_user:
            h_end_idx = clean_user.find(_HISTORY_END_MARKER)
            history_body = clean_user[len(_HISTORY_START_MARKER):h_end_idx].strip()
            task_user = clean_user[h_end_idx + len(_HISTORY_END_MARKER):].lstrip()
            if history_body:
                turn_chunks = [t.strip() for t in _TURN_SPLIT_RE.split(history_body) if t.strip()]
                for idx, chunk in enumerate(turn_chunks):
                    if idx == 0:
                        prefix_segments.append(f"{_HISTORY_START_MARKER}{chunk}\n")
                    else:
                        prefix_segments.append(f"{chunk}\n")
                suffix_str = (
                    f"--- END CONVERSATION HISTORY ---\n\n{task_sys}\n<|im_end|>\n"
                    f"<|im_start|>user\n{task_user}\n<|im_end|>\n"
                    f"{assistant_hdr}"
                )
                return prefix_segments, suffix_str
            else:
                clean_user = task_user

        suffix_str = (
            f"\n{task_sys}\n<|im_end|>\n"
            f"<|im_start|>user\n{clean_user}\n<|im_end|>\n"
            f"{assistant_hdr}"
        )
        return prefix_segments, suffix_str

    def _prepare_cached_prompt_tokens(
        self,
        prefix_segments: List[str],
        suffix_str: str,
    ) -> Tuple[List[int], int]:
        """
        Restores or advances the sequence-state prefix checkpoint to match `prefix_segments`,
        returning `(full_prompt_tokens, cached_prefix_token_count)`.
        """
        if not prefix_segments:
            tokens = list(self._tokenize_segment(suffix_str, is_first=True))
            self._llm.reset()
            return tokens, 0

        chain: List[Tuple[int, ...]] = []
        acc: Tuple[int, ...] = ()
        for idx, seg in enumerate(prefix_segments):
            acc = acc + self._tokenize_segment(seg, is_first=(idx == 0))
            chain.append(acc)

        target_prefix = chain[-1]
        suffix_tokens = list(self._tokenize_segment(suffix_str, is_first=False))
        full_tokens = list(target_prefix) + suffix_tokens

        if len(full_tokens) >= self.active_n_ctx:
            self._llm.reset()
            return full_tokens[: self.active_n_ctx - 64], 0

        # Check if GPU state is already sitting at exact target_prefix
        n_target = len(target_prefix)
        if (
            self._llm.n_tokens == n_target
            and tuple(self._llm.input_ids[:n_target].tolist()) == target_prefix
        ):
            self._llm.input_ids[n_target:] = -1
            return full_tokens, n_target

        # Find longest cached prefix in chain
        hit_idx = -1
        for idx in range(len(chain) - 1, -1, -1):
            if chain[idx] in self._prefix_state_cache:
                hit_idx = idx
                break

        if hit_idx >= 0:
            hit_key = chain[hit_idx]
            state_bytes = self._prefix_state_cache[hit_key]
            if hit_key != self._pinned_base_key:
                self._prefix_state_cache.move_to_end(hit_key)

            if self._restore_seq0_state(hit_key, state_bytes):
                cached_tokens = len(hit_key)
                if hit_idx < len(chain) - 1:
                    delta_tokens = list(target_prefix[len(hit_key):])
                    if delta_tokens:
                        self._llm.eval(delta_tokens)
                        self._llm.input_ids[len(target_prefix):] = -1
                        self._store_prefix_checkpoint(target_prefix, self._save_seq0_state(), pin=False)
                return full_tokens, cached_tokens

        # Cold cache: evaluate base memory prefix P_0 and snapshot it
        self._llm.reset()
        base_key = chain[0]
        self._llm.eval(list(base_key))
        self._llm.input_ids[len(base_key):] = -1
        self._store_prefix_checkpoint(base_key, self._save_seq0_state(), pin=True)

        if len(chain) > 1:
            delta_tokens = list(target_prefix[len(base_key):])
            if delta_tokens:
                self._llm.eval(delta_tokens)
                self._llm.input_ids[len(target_prefix):] = -1
                self._store_prefix_checkpoint(target_prefix, self._save_seq0_state(), pin=False)

        return full_tokens, 0

    # ── BaseLLMEngine hooks ──────────────────────────────────────────────────

    def _build_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        disable_thinking: bool,
        response_prefix: str = "",
    ) -> str:
        """Assemble full ChatML prompt string (fallback / inspection helper)."""
        prefix_segs, suffix_str = self._split_prompt_segments(
            system_prompt, user_prompt, disable_thinking, response_prefix
        )
        return "".join(prefix_segs) + suffix_str

    def _sampling_params(self, disable_thinking: bool) -> Dict[str, Any]:
        temp = 0.7 if disable_thinking else 0.6
        top_p = 0.80 if disable_thinking else 0.95
        return {
            "temperature": temp,
            "top_p": top_p,
            "stop": ["<|im_end|>", "<|endoftext|>"],
        }

    def _record_perf(self, total_prompt_tokens: int, cached_tokens: int, t_prep_ms: float) -> None:
        """Reads native llama.cpp context performance counters into self.last_hop_telemetry."""
        import llama_cpp
        try:
            perf = llama_cpp.llama_perf_context(self._llm._ctx.ctx)
            n_p_eval = int(perf.n_p_eval)
            n_eval = int(perf.n_eval)
            t_p_eval_ms = float(perf.t_p_eval_ms)
            t_eval_ms = float(perf.t_eval_ms)
        except Exception:
            n_p_eval = max(0, total_prompt_tokens - cached_tokens)
            n_eval = 0
            t_p_eval_ms = 0.0
            t_eval_ms = 0.0

        prefill_ms = round(t_prep_ms + t_p_eval_ms, 1)
        decode_ms = round(t_eval_ms, 1)
        tok_s = round(n_eval / (t_eval_ms / 1000.0), 2) if t_eval_ms > 0 else 0.0
        self.last_hop_telemetry = {
            "prompt_tokens": total_prompt_tokens,
            "cached_prefix_tokens": cached_tokens,
            "eval_prompt_tokens": max(0, total_prompt_tokens - cached_tokens),
            "llama_n_p_eval": n_p_eval,
            "completion_tokens": n_eval,
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "tok_per_s": tok_s,
        }

    def _generate_raw(
        self,
        system_prompt: str,
        user_prompt: str,
        disable_thinking: bool = False,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        response_prefix: str = "",
    ) -> str:
        """
        Execute completion using sequence-state prefix checkpointing and native Qwen3.8 ChatML.
        """
        import llama_cpp

        t_prep_start = time.perf_counter()
        prefix_segs, suffix_str = self._split_prompt_segments(
            system_prompt, user_prompt, disable_thinking, response_prefix
        )
        prompt_tokens, cached_tokens = self._prepare_cached_prompt_tokens(prefix_segs, suffix_str)
        t_prep_ms = (time.perf_counter() - t_prep_start) * 1000.0

        try:
            llama_cpp.llama_perf_context_reset(self._llm._ctx.ctx)
        except Exception:
            pass

        response = self._llm(
            prompt_tokens,
            max_tokens=max_tokens,
            echo=False,
            **self._sampling_params(disable_thinking),
        )
        self._record_perf(len(prompt_tokens), cached_tokens, t_prep_ms)
        return response["choices"][0]["text"]

    def _generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        disable_thinking: bool = False,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        response_prefix: str = "",
    ):
        """
        Token-level streaming variant of _generate_raw using sequence-state prefix checkpointing.
        If response_prefix is provided, yields response_prefix first so callers immediately
        receive the pre-filled JSON header.
        """
        import llama_cpp

        t_prep_start = time.perf_counter()
        prefix_segs, suffix_str = self._split_prompt_segments(
            system_prompt, user_prompt, disable_thinking, response_prefix
        )
        prompt_tokens, cached_tokens = self._prepare_cached_prompt_tokens(prefix_segs, suffix_str)
        t_prep_ms = (time.perf_counter() - t_prep_start) * 1000.0

        try:
            llama_cpp.llama_perf_context_reset(self._llm._ctx.ctx)
        except Exception:
            pass

        if response_prefix:
            yield response_prefix

        stream = self._llm(
            prompt_tokens,
            max_tokens=max_tokens,
            echo=False,
            stream=True,
            **self._sampling_params(disable_thinking),
        )
        try:
            for chunk in stream:
                text = chunk["choices"][0]["text"]
                if text:
                    yield text
        finally:
            self._record_perf(len(prompt_tokens), cached_tokens, t_prep_ms)


# Alias for compatibility with direct imports
Qwen3827BEngine = Qwen3_8_27BEngine
