"""
core/ocr/glm_ocr.py — GLM-OCR Engine (Hugging Face transformers Backend).

Features:
  1. Native Hugging Face Loading:
     - AutoProcessor + AutoModelForImageTextToText with trust_remote_code=True,
       torch_dtype=float16, device_map="cuda".
  2. Lightweight Orientation Correction:
     - ImageOps.exif_transpose rotates sideways pages (90/180/270 degrees) upright.
  3. Standard Native Chat Template Inference:
     - Single greedy generation pass (do_sample=False) with dynamic token cap
       max_tokens = min(config.OCR_MAX_NEW_TOKENS, 2048).
  4. Memory Lifecycle Support:
     - close() releases model/processor references and reclaims VRAM.

Model: zai-org/GLM-OCR (MIT License)
"""

import gc
import logging
import os

import torch
from PIL import Image, ImageOps
from transformers import AutoModelForImageTextToText, AutoProcessor

from core.base import BaseOCREngine

logger = logging.getLogger(__name__)


class GLMOCREngine(BaseOCREngine):
    """
    High-throughput OCR engine backed by GLM-OCR (Hugging Face transformers).
    Subclasses BaseOCREngine for seamless single-residency lifecycle management.
    """

    MODEL_ID = os.getenv("GLM_OCR_MODEL_ID", "zai-org/GLM-OCR")
    ENGINE_NAME = "GLM-OCR"

    def __init__(self) -> None:
        self._processor = None
        self._model = None

    def load(self, cache_dir: str) -> None:
        """Load GLM-OCR model using the Hugging Face transformers pipeline."""
        if self._model is not None and self._processor is not None:
            logger.info("⚡ GLM-OCR (HF) is already loaded, skipping reload.")
            return

        logger.info(f"⏳ Loading '{self.MODEL_ID}' via transformers into GPU memory...")
        try:
            self._processor = AutoProcessor.from_pretrained(
                self.MODEL_ID,
                trust_remote_code=True,
                cache_dir=cache_dir,
            )
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.MODEL_ID,
                torch_dtype=torch.float16,
                device_map="cuda",
                trust_remote_code=True,
                cache_dir=cache_dir,
            ).eval()
            logger.info("🚀 GLM-OCR loaded successfully with Hugging Face execution core!")
        except Exception as e:
            logger.error(f"❌ CRITICAL ERROR: Failed to load GLM-OCR via transformers. Details: {e}")
            raise RuntimeError(f"Failed to load GLM-OCR via transformers: {e}") from e

    def _correct_orientation(self, image_path: str) -> Image.Image:
        """
        Lightweight orientation check: applies EXIF transpose to rotate sideways
        pages (90/180/270 degrees) to standard upright orientation without
        modifying pixel values or destroying handwriting/signature details.
        """
        try:
            raw_img = Image.open(image_path)
            corrected_img = ImageOps.exif_transpose(raw_img)
            return corrected_img.convert("RGB")
        except Exception as e:
            logger.warning(f"⚠️ Orientation check warning for {image_path}: {e}")
            return Image.open(image_path).convert("RGB")

    def _run_inference(self, image_path: str, output_folder: str = "") -> str:
        """
        Executes OCR inference with:
          1. Native image handling (no destructive binarization/filtering).
          2. EXIF orientation auto-correction.
          3. Standard native chat-template generation (greedy, do_sample=False).
        """
        from server.config import OCR_MAX_NEW_TOKENS

        upright_img = self._correct_orientation(image_path)
        max_toks = min(OCR_MAX_NEW_TOKENS, 2048)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": upright_img},
                    {"type": "text", "text": "Text Recognition:"},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_toks,
                do_sample=False,
                temperature=None,
                pad_token_id=(
                    self._processor.tokenizer.pad_token_id
                    or self._processor.tokenizer.eos_token_id
                ),
            )

        input_len = inputs["input_ids"].shape[1]
        output_ids = generated_ids[:, input_len:]
        text_output = self._processor.decode(output_ids[0], skip_special_tokens=True).strip()
        return text_output if text_output else "Error: GLM-OCR returned empty output."

    def close(self) -> None:
        """Releases GLM-OCR model and processor from GPU memory and flushes CUDA cache."""
        logger.info("🗑️ Unloading GLM-OCR model and releasing GPU VRAM...")
        self._model = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("✅ GLM-OCR unloaded successfully. VRAM reclaimed.")