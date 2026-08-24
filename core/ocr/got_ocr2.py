"""
core/ocr/got_ocr2.py — GOT-OCR2_0 engine implementation.

Lightweight, end-to-end OCR model (0.5B params).
Runs in FP16 precision using ~1.5GB VRAM.

Model: ucaslcl/GOT-OCR2_0
"""

import logging
import sys
import torch
from transformers import AutoModel, AutoTokenizer

from core.base import BaseOCREngine

logger = logging.getLogger(__name__)


class GOTOCR2Engine(BaseOCREngine):
    """
    OCR engine backed by GOT-OCR2_0 (0.5B params).
    Uses model.chat(tokenizer, image_path, ocr_type='format') for structured OCR.
    """

    MODEL_ID = "ucaslcl/GOT-OCR2_0"
    ENGINE_NAME = "GOT-OCR2_0"

    def __init__(self):
        self._model = None
        self._tokenizer = None

    def load(self, cache_dir: str) -> None:
        """Load GOT-OCR2_0 model and tokenizer."""
        if self._model is not None:
            logger.info(" GOT-OCR2_0 is already loaded, skipping reload.")
            return

        logger.info(f"⏳ Loading '{self.MODEL_ID}' into GPU memory...")

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL_ID,
                trust_remote_code=True,
                cache_dir=cache_dir,
            )

            device_target = "cuda" if torch.cuda.is_available() else "cpu"

            self._model = AutoModel.from_pretrained(
                self.MODEL_ID,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map=device_target,
                cache_dir=cache_dir,
            ).eval()

        except Exception as e:
            logger.error(f" CRITICAL ERROR: Failed to load GOT-OCR2_0. Details: {e}")
            raise RuntimeError(f"Failed to load GOT-OCR2_0: {e}") from e

        logger.info(" GOT-OCR2_0 loaded successfully!")

    def _run_inference(self, image_path: str, output_folder: str = "") -> str:
        """Run inference using model.chat() with layout formatting."""
        with torch.no_grad():
            res = self._model.chat(self._tokenizer, image_path, ocr_type="format")
        return res if res else "Error: GOT-OCR2_0 returned empty output."
