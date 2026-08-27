"""
core/base.py — Abstract base classes for pluggable OCR and LLM core.

Defines the contracts that every OCR and LLM model implementation must follow.
Adding a new model to the pipeline = subclassing the appropriate base + registering it.
"""

import json
import logging
import os
import re
from abc import ABC, abstractmethod
try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None

from server.config import DEFAULT_MAX_TOKENS

logger = logging.getLogger(__name__)
_TMP_PDF_IMAGE_PATTERN = "temp_cert_page_{}.jpg"

#: Reasoning-trace markers emitted by models before JSON answers.
_THINKING_BLOCK_RE = re.compile(r"\s*<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL)
_COT_BLOCK_RE = re.compile(r"\s*thinking.*?response\s*", re.DOTALL)
_CHANNEL_BLOCK_RE = re.compile(r"\s*<\|channel\>.*?<channel\|\>\s*", re.DOTALL)
_STRAY_THINK_TAG_RE = re.compile(r"\s*</?think>", re.DOTALL)
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*$|^```\s*$", re.MULTILINE)

#: Balanced-brace JSON object matcher (handles nested braces up to 3 levels).
_BALANCED_JSON_RE = re.compile(
    r"\{(?:[^{}]|\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})*\}"
)


def strip_reasoning_traces(raw_content: str) -> str:
    """
    Remove reasoning/thinking traces unconditionally from raw completion output.
    """
    content = raw_content.strip()
    content = _THINKING_BLOCK_RE.sub("", content).strip()
    content = _COT_BLOCK_RE.sub("", content).strip()
    content = _CHANNEL_BLOCK_RE.sub("", content).strip()
    content = _STRAY_THINK_TAG_RE.sub("", content).strip()
    content = _FENCE_RE.sub("", content).strip()
    return content


def extract_json(raw_content: str) -> str:
    """
    Parse or extract a strict JSON string from raw model output.

    Ladder (in order):
      1. Scan raw content for a balanced-brace JSON object that parses.
      2. Direct json.loads() on raw content.
      3. Scrub reasoning traces, then re-run direct parse & balanced scan.
      4. Fallback to "{}" if unparseable.
    """
    if not raw_content:
        return "{}"

    def _first_parseable(candidate_text: str) -> str | None:
        for json_match in _BALANCED_JSON_RE.finditer(candidate_text):
            candidate = json_match.group(0)
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue
        return None

    raw_hit = _first_parseable(raw_content)
    if raw_hit is not None:
        return raw_hit

    try:
        json.loads(raw_content)
        return raw_content
    except json.JSONDecodeError:
        pass

    content = strip_reasoning_traces(raw_content)
    if not content:
        return "{}"

    try:
        json.loads(content)
        return content
    except json.JSONDecodeError:
        pass

    scrubbed_hit = _first_parseable(content)
    if scrubbed_hit is not None:
        return scrubbed_hit

    return "{}"


class BaseOCREngine(ABC):
    """
    Abstract contract for all OCR core.

    Every OCR implementation (DeepSeek-OCR-2, GLM-OCR, GOT-OCR2, etc.)
    must subclass this and implement load() and _run_inference().
    """

    ENGINE_NAME: str = "OCR"

    @abstractmethod
    def load(self, cache_dir: str) -> None:
        """
        Load model weights into GPU/CPU memory.

        Args:
            cache_dir: Local directory for caching downloaded weights.
        """
        ...

    @abstractmethod
    def _run_inference(self, image_path: str, output_folder: str = "") -> str:
        """
        Model-specific single-image inference execution.

        Args:
            image_path:    Absolute path to the target image file.
            output_folder: Folder for intermediate model outputs (if applicable).

        Returns:
            Extracted layout Markdown text string.
        """
        ...

    def process_document(self, file_path: str, output_folder: str) -> str:
        """
        Run OCR inference on a document (image or multi-page PDF).

        Template method: resolves PDF/image paths, iterates through pages,
        calls model-specific _run_inference(), and injects <Page X> delimiters.

        Raises RuntimeError if any page fails so callers never treat partial or
        error-marked OCR output as a successful extraction (which previously led
        to empty records being persisted and error text being cached).

        Args:
            file_path:     Absolute path to the input document.
            output_folder: Folder for temporary intermediate files.

        Returns:
            Layout-preserved Markdown string with <Page X> tags injected.
        """
        if not os.path.exists(file_path):
            raise RuntimeError(f"File not found: '{file_path}'.")

        image_paths = self._resolve_image_paths(file_path, output_folder)
        if not image_paths:
            raise RuntimeError("Could not extract any images from the document.")

        extracted_pages = []
        failed_pages = []
        engine_name = getattr(self, "ENGINE_NAME", self.__class__.__name__)

        for i, target_image_path in enumerate(image_paths):
            page_num = i + 1
            logger.info(f"  Running {engine_name} inference on page {page_num}/{len(image_paths)}...")
            try:
                raw_output = self._run_inference(target_image_path, output_folder)
                page_content = f"\n\n<Page {page_num}>\n\n{raw_output}"
                extracted_pages.append(page_content)
            except Exception as e:
                error_msg = f"Error processing page {page_num}: {e}"
                logger.error(f"  {error_msg}")
                failed_pages.append(error_msg)

        if failed_pages:
            raise RuntimeError("; ".join(failed_pages))

        return "".join(extracted_pages).strip()

    def _resolve_image_paths(self, file_path: str, output_folder: str) -> list[str]:
        """
        Helper method to resolve input file paths: converts PDF pages to temporary
        JPEG images, or returns the image path directly for standard image files.
        """
        if file_path.lower().endswith(".pdf"):
            logger.info(f" Converting PDF to high-resolution images (200 DPI): {os.path.basename(file_path)}")
            if convert_from_path is None:
                raise ImportError(
                    "pdf2image is not installed. Run: pip install pdf2image && apt-get install -y poppler-utils"
                )
            try:
                pages = convert_from_path(file_path, dpi=200)
            except Exception as e:
                logger.error(f" Error converting PDF: {e}")
                return []

            if output_folder:
                os.makedirs(output_folder, exist_ok=True)
            image_paths = []
            for i, page in enumerate(pages):
                tmp_path = os.path.join(output_folder, _TMP_PDF_IMAGE_PATTERN.format(i + 1))
                page.save(tmp_path, "JPEG", quality=95)
                image_paths.append(tmp_path)

            return image_paths

        logger.info(f"  Using image: {os.path.basename(file_path)}")
        return [file_path]


class BaseLLMEngine(ABC):
    """
    Abstract contract for all LLM core.

    Every LLM implementation must subclass this and implement:
      1. load(cache_dir: str) -> None
      2. _generate_raw(system_prompt, user_prompt, disable_thinking, max_tokens) -> str

    generate_json() is a concrete template method implemented here in the base class:
    it executes _generate_raw() and automatically scrubs reasoning traces / extracts strict JSON.
    """

    @abstractmethod
    def load(self, cache_dir: str) -> None:
        """
        Load model weights into GPU/CPU memory.

        Args:
            cache_dir: Local directory for caching downloaded weights.
        """
        ...

    @abstractmethod
    def _generate_raw(
        self,
        system_prompt: str,
        user_prompt: str,
        disable_thinking: bool = False,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """
        Execute model-specific prompt formatting, sampling parameter selection,
        and raw inference completion.

        Args:
            system_prompt:    System instructions.
            user_prompt:      User query payload.
            disable_thinking: True for fast non-thinking mode; False for deep reasoning mode.
            max_tokens:       Maximum completion tokens to generate.

        Returns:
            The raw text output from the model (may contain reasoning traces).
        """
        ...

    def generate_json(
        self,
        system_prompt: str = "",
        user_prompt: str = "",
        disable_thinking: bool = False,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """
        Template method: executes _generate_raw() and extracts strict JSON output.

        Args:
            system_prompt: System instructions for the completion.
            user_prompt:   User-facing query / instruction payload.
            disable_thinking: True for fast non-thinking mode; False for deep reasoning mode.
            max_tokens: Maximum completion tokens to generate.

        Returns:
            A raw JSON string parseable by json.loads().
        """
        if getattr(self, "_llm", None) is None:
            raise RuntimeError("Model not loaded. Call load() before generate_json().")

        if disable_thinking:
            logger.info(
                f" Running {self.__class__.__name__} in Fast Non-Thinking Mode "
                f"(max_tokens={max_tokens})..."
            )
        else:
            logger.info(
                f" Running {self.__class__.__name__} in Deep Thinking Mode "
                f"(max_tokens={max_tokens})..."
            )

        try:
            raw_content = self._generate_raw(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                disable_thinking=disable_thinking,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            logger.error(f" {self.__class__.__name__} completion failed: {exc}")
            return "{}"

        logger.debug(f"Raw model output (first 500 chars): {raw_content[:500]}")
        return extract_json(raw_content)

    def generate_stream(
        self,
        system_prompt: str = "",
        user_prompt: str = "",
        disable_thinking: bool = False,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        """
        Template method: streams the raw model output token-by-token.

        Delegates to the model-specific _generate_stream() hook. Engines that do not
        override the hook fall back to yielding the complete _generate_raw() result in
        a single chunk, so streaming is always safe to call.

        Args:
            system_prompt: System instructions for the completion.
            user_prompt:   User-facing query / instruction payload.
            disable_thinking: True for fast non-thinking mode; False for deep reasoning mode.
            max_tokens: Maximum completion tokens to generate.

        Yields:
            str: Incremental raw text chunks from the model.
        """
        if getattr(self, "_llm", None) is None:
            raise RuntimeError("Model not loaded. Call load() before generate_stream().")

        logger.info(
            f" Streaming {self.__class__.__name__} generation "
            f"(disable_thinking={disable_thinking}, max_tokens={max_tokens})..."
        )
        yield from self._generate_stream(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            disable_thinking=disable_thinking,
            max_tokens=max_tokens,
        )

    def _generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        disable_thinking: bool = False,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        """
        Model-specific streaming hook. Default implementation yields the complete
        non-streaming result in one chunk. Engines that support incremental decoding
        (e.g. llama-cpp-python `stream=True`) override this to yield real tokens.
        """
        yield self._generate_raw(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            disable_thinking=disable_thinking,
            max_tokens=max_tokens,
        )

