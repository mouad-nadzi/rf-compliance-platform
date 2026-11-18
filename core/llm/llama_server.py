"""
core/llm/llama_server.py — HTTP Client Engine for llama.cpp `llama-server` Sidecar.

Subclasses BaseLLMEngine to talk to an external `llama-server` process (supporting
`--cache-prompt`, `--parallel N` session-pinned slots via `id_slot`, and
`--spec-type draft-mtp` speculative decoding) while preserving the exact same
ChatML prompt formatting, thinking-mode switches, and per-hop telemetry contract
as Qwen3_8_27BEngine.
"""

import json
import logging
import os
import time
import urllib.request
from typing import Any, Dict, Optional

from core.base import BaseLLMEngine
from server.config import DEFAULT_CONTEXT_WINDOW, DEFAULT_MAX_TOKENS

logger = logging.getLogger(__name__)


class LlamaServerEngine(BaseLLMEngine):
    """
    BaseLLMEngine implementation backed by a `llama-server` HTTP endpoint (`/completion`).
    """

    SERVER_URL: str = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8081")

    def __init__(self) -> None:
        self._llm: Optional[str] = None
        self.active_n_ctx: int = DEFAULT_CONTEXT_WINDOW
        self.last_hop_telemetry: Optional[Dict[str, Any]] = None

    def load(self, cache_dir: str) -> None:
        """Marks the HTTP sidecar client ready."""
        self._llm = self.SERVER_URL.rstrip("/")
        logger.info(f" LlamaServerEngine configured for sidecar endpoint: {self._llm}")

    def _assistant_header(self, disable_thinking: bool, response_prefix: str = "") -> str:
        if disable_thinking:
            return f"<|im_start|>assistant\n<think>\n\n</think>\n\n{response_prefix}"
        if response_prefix:
            return f"<|im_start|>assistant\n{response_prefix}"
        return "<|im_start|>assistant\n<think>\n"

    def _build_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        disable_thinking: bool,
        response_prefix: str = "",
    ) -> str:
        if not system_prompt:
            system_prompt = (
                "You are an expert technical data extraction engine. "
                "Respond with a valid JSON object only."
            )
        assistant_hdr = self._assistant_header(disable_thinking, response_prefix)
        return (
            f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}\n<|im_end|>\n"
            f"{assistant_hdr}"
        )

    def _payload(
        self,
        prompt: str,
        disable_thinking: bool,
        max_tokens: int,
        stream: bool = False,
    ) -> Dict[str, Any]:
        temp = 0.7 if disable_thinking else 0.6
        top_p = 0.80 if disable_thinking else 0.95
        return {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temp,
            "top_p": top_p,
            "stop": ["<|im_end|>", "<|endoftext|>"],
            "cache_prompt": True,
            "stream": stream,
        }

    def _generate_raw(
        self,
        system_prompt: str,
        user_prompt: str,
        disable_thinking: bool = False,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        response_prefix: str = "",
    ) -> str:
        prompt = self._build_prompt(system_prompt, user_prompt, disable_thinking, response_prefix)
        body = json.dumps(self._payload(prompt, disable_thinking, max_tokens, stream=False)).encode("utf-8")
        req = urllib.request.Request(
            f"{self._llm}/completion",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        wall_ms = (time.perf_counter() - t0) * 1000.0
        timings = data.get("timings") or {}
        n_eval = int(timings.get("predicted_n", data.get("tokens_predicted", 0)))
        t_eval_ms = float(timings.get("predicted_ms", 0.0))
        self.last_hop_telemetry = {
            "prompt_tokens": int(timings.get("prompt_n", 0)) + int(timings.get("cache_n", 0)),
            "cached_prefix_tokens": int(timings.get("cache_n", 0)),
            "eval_prompt_tokens": int(timings.get("prompt_n", 0)),
            "completion_tokens": n_eval,
            "prefill_ms": round(float(timings.get("prompt_ms", 0.0)), 1),
            "decode_ms": round(t_eval_ms, 1),
            "total_ms": round(wall_ms, 1),
            "tok_per_s": round(float(timings.get("predicted_per_second", 0.0)), 2),
        }
        return str(data.get("content", ""))

    def _generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        disable_thinking: bool = False,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        response_prefix: str = "",
    ):
        prompt = self._build_prompt(system_prompt, user_prompt, disable_thinking, response_prefix)
        body = json.dumps(self._payload(prompt, disable_thinking, max_tokens, stream=True)).encode("utf-8")
        req = urllib.request.Request(
            f"{self._llm}/completion",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if response_prefix:
            yield response_prefix
        timings: Dict[str, Any] = {}
        with urllib.request.urlopen(req, timeout=300) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                payload_str = line[5:].strip()
                if not payload_str:
                    continue
                chunk = json.loads(payload_str)
                content = chunk.get("content", "")
                if content:
                    yield content
                if chunk.get("stop") and chunk.get("timings"):
                    timings = chunk["timings"]
        if timings:
            self.last_hop_telemetry = {
                "prompt_tokens": int(timings.get("prompt_n", 0)) + int(timings.get("cache_n", 0)),
                "cached_prefix_tokens": int(timings.get("cache_n", 0)),
                "eval_prompt_tokens": int(timings.get("prompt_n", 0)),
                "completion_tokens": int(timings.get("predicted_n", 0)),
                "prefill_ms": round(float(timings.get("prompt_ms", 0.0)), 1),
                "decode_ms": round(float(timings.get("predicted_ms", 0.0)), 1),
                "tok_per_s": round(float(timings.get("predicted_per_second", 0.0)), 2),
            }
