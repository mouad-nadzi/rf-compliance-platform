"""
core/ocr/deepseek_ocr2.py — DeepSeek-OCR-2 engine implementation.

Wraps the existing DeepSeek-OCR-2 model loading and inference logic
inside a BaseOCREngine subclass for registry compatibility.

This is the original OCR engine (3B parameters, full FP16 — no quantization).
"""

import io
import importlib
import logging
import os
import sys
from contextlib import redirect_stdout

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from core.base import BaseOCREngine

logger = logging.getLogger(__name__)

# The grounding prompt understood by DeepSeek-OCR-2
_OCR_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."

# transformers 5.x instantiates remote configs without running the custom
# __init__ body, so only keys present in config.json land in __dict__. Class
# level defaults below supply every remaining attribute the remote model code
# reads. Values mirror DeepseekV2Config.__init__ defaults.
_DEFAULT_CONFIG_PATCHES = {
    "hidden_act": "silu",
    "attention_bias": False,
    "attention_dropout": 0.0,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "rope_scaling": None,
    "initializer_range": 0.02,
    "pad_token_id": None,
    "pretraining_tp": 1,
    "use_cache": True,
    "aux_loss_alpha": 0.001,
    "ep_size": 1,
    "moe_layer_freq": 1,
    "norm_topk_prob": False,
    "routed_scaling_factor": 1.0,
    "scoring_func": "softmax",
    "seq_aux": True,
}


def _patch_config_class_defaults(cache_dir: str | None = None) -> bool:
    """Attach missing model-read attributes as defaults on the remote config class."""
    import glob as _glob

    mod = None
    for m in list(sys.modules.values()):
        name = getattr(m, "__name__", "")
        if name.startswith("transformers_modules.") and hasattr(m, "DeepseekOCR2Config"):
            mod = m
            break
    if mod is None and cache_dir:
        modules_root = os.path.join(cache_dir, "modules")
        if os.path.isdir(modules_root) and modules_root not in sys.path:
            sys.path.insert(0, modules_root)
        for py in _glob.glob(
            os.path.join(
                modules_root,
                "transformers_modules",
                "deepseek_hyphen_ai",
                "DeepSeek_hyphen_OCR_hyphen_2",
                "*",
                "modeling_deepseekocr2.py",
            )
        ):
            rel = os.path.relpath(py, modules_root)
            modname = rel[:-3].replace(os.sep, ".")
            try:
                mod = importlib.import_module(modname)
            except Exception:
                continue
            if hasattr(mod, "DeepseekOCR2Config"):
                break
    if mod is None:
        return False
    cls = mod.DeepseekOCR2Config
    for k, v in _DEFAULT_CONFIG_PATCHES.items():
        if not hasattr(cls, k):
            setattr(cls, k, v)
    from transformers import GenerationMixin

    for mname in ("DeepseekOCR2ForCausalLM", "DeepseekV2ForCausalLM"):
        mc = getattr(mod, mname, None)
        if mc is not None and not hasattr(mc, "generate"):
            try:
                mc.__bases__ = (GenerationMixin,) + mc.__bases__
            except TypeError:
                pass
    return True


def _install_transformers_compat():
    """Bridge DeepSeek-OCR-2's remote code to transformers 5.x.

    The model's remote ``modeling_deepseekv2.py`` was written against a
    transformers ~4.4x/4.5x API that is no longer present in transformers 5.x:

    * ``LlamaFlashAttention2`` was removed from
      ``transformers.models.llama.modeling_llama`` (flash attention is now
      folded into ``LlamaAttention``).
    * ``LlamaAttention.forward`` now requires a precomputed
      ``position_embeddings`` tuple instead of ``position_ids``, and returns a
      2-tuple ``(attn_output, attn_weights)`` instead of the legacy 3-tuple
      ``(attn_output, attn_weights, present_key_value)``.

    The deepseek decoder layer calls ``self_attn`` with the legacy signature
    (``position_ids``, ``past_key_value``, ...) and unpacks a 3-tuple, so we
    wrap the installed ``LlamaAttention`` to bridge both APIs. This must run
    *before* ``AutoTokenizer/AutoConfig`` triggers the remote-code import.
    """
    import transformers.utils.import_utils as import_utils_mod
    if not hasattr(import_utils_mod, "is_torch_fx_available"):
        import_utils_mod.is_torch_fx_available = lambda: False

    import transformers.models.llama.modeling_llama as llama_mod

    from transformers.models.llama.modeling_llama import LlamaAttention as NewLlamaAttention

    class LegacyLlamaAttention(NewLlamaAttention):
        def __init__(self, config, layer_idx=None):
            if not hasattr(config, "attention_dropout"):
                config.attention_dropout = 0.0
            if not hasattr(config, "attention_bias"):
                config.attention_bias = False
            if not hasattr(config, "rope_parameters"):
                config.rope_parameters = {
                    "rope_type": "default",
                    "rope_theta": getattr(config, "rope_theta", 10000.0),
                }
            super().__init__(config, layer_idx)
            if not hasattr(self, "rotary_emb"):
                from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

                self.rotary_emb = LlamaRotaryEmbedding(self.config)

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            position_embeddings=None,
            **kwargs,
        ):
            if position_embeddings is None:
                cos, sin = self.rotary_emb(hidden_states, position_ids)
                position_embeddings = (cos, sin)
            attn_output, attn_weights = super().forward(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_value,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            present_key_value = past_key_value if use_cache else None
            return attn_output, attn_weights, present_key_value

    llama_mod.LlamaAttention = LegacyLlamaAttention
    llama_mod.LlamaFlashAttention2 = LegacyLlamaAttention

    from transformers.cache_utils import DynamicCache

    # transformers 5.x removed DynamicCache.seen_tokens (and _seen_tokens);
    # the deepseek generation code still reads `past_key_values.seen_tokens`
    # for the prompt length. Delegate to get_seq_length().
    if not hasattr(DynamicCache, "seen_tokens"):

        def _get_seen_tokens(self):
            return self.get_seq_length()

        def _set_seen_tokens(self, value):
            pass

        DynamicCache.seen_tokens = property(_get_seen_tokens, _set_seen_tokens)

    # DeepseekOCR2Model.forward scatters the vision features into the fp16
    # ``inputs_embeds`` via ``masked_scatter_``. The features are built with
    # ``torch.cat([..., self.view_seperator[None, :]])`` where ``view_seperator``
    # is created fp32, so ``cat`` promotes the source to fp32 and the op raises
    # "masked_scatter_: expected self and source to have same dtypes". Cast the
    # source to the target dtype so the scatter always succeeds.
    _orig_masked_scatter_ = torch.Tensor.masked_scatter_

    def _masked_scatter_same_dtype(self, mask, source):
        if self.dtype != source.dtype:
            source = source.to(self.dtype)
        return _orig_masked_scatter_(self, mask, source)

    torch.Tensor.masked_scatter_ = _masked_scatter_same_dtype

    llama_mod._deepseek_ocr2_compat_installed = True
    logger.info("🩹 Installed DeepSeek-OCR-2 ↔ transformers 5.x compatibility shim (LlamaAttention + import_utils).")
    _patch_config_class_defaults()


class DeepSeekOCR2Engine(BaseOCREngine):
    """
    OCR engine backed by the DeepSeek-OCR-2 vision-language model.

    Loaded in full FP16 precision (no bitsandbytes quantization) for maximum
    accuracy, mirroring the pure-HF GLM-OCR backend. The 3B model (~6 GB FP16)
    fits comfortably in the T4's 15.36 GB when resident alone (single-residency
    sequential lifecycle guarantees this).
    """

    MODEL_ID   = "deepseek-ai/DeepSeek-OCR-2"
    ENGINE_NAME = "DeepSeek-OCR-2"
    BASE_SIZE   = 1024
    IMAGE_SIZE  = 768
    CROP_MODE   = True

    def __init__(self):
        self._model = None
        self._tokenizer = None

    # ── BaseOCREngine interface ───────────────────────────────────────────

    def load(self, cache_dir: str) -> None:
        """Load the DeepSeek-OCR-2 model in full FP16 (no quantization)."""
        if self._model is not None:
            logger.info("⚡ DeepSeek-OCR-2 is already loaded, skipping reload.")
            return

        _install_transformers_compat()

        logger.info(f"⏳ Loading '{self.MODEL_ID}' into GPU memory (full FP16, no quantization)...")

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL_ID,
                trust_remote_code=True,
                cache_dir=cache_dir,
            )

            if not _patch_config_class_defaults(cache_dir):
                AutoConfig.from_pretrained(
                    self.MODEL_ID, trust_remote_code=True, cache_dir=cache_dir
                )
                _patch_config_class_defaults(cache_dir)

            self._model = AutoModel.from_pretrained(
                self.MODEL_ID,
                trust_remote_code=True,
                device_map="cuda",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
                cache_dir=cache_dir,
            ).eval()

        except Exception as e:
            logger.error(f"❌ CRITICAL ERROR: Unexpected error loading DeepSeek-OCR-2. Details: {e}")
            raise RuntimeError(f"DeepSeek-OCR-2 failed to load: {e}") from e

        logger.info("✅ DeepSeek-OCR-2 loaded successfully (full FP16)!")

    # ── Private helpers ───────────────────────────────────────────────────

    def _run_inference(self, image_path: str, output_folder: str = "") -> str:
        """Execute model inference, capturing stdout output."""

        if hasattr(self._model, "generation_config"):
            self._model.generation_config.max_new_tokens = 800
            self._model.generation_config.repetition_penalty = 1.5

        buffer = io.StringIO()
        with torch.no_grad():
            with redirect_stdout(buffer):
                self._model.infer(
                    self._tokenizer,
                    prompt=_OCR_PROMPT,
                    image_file=image_path,
                    output_path=output_folder,
                    base_size=self.BASE_SIZE,
                    image_size=self.IMAGE_SIZE,
                    crop_mode=self.CROP_MODE,
                )
        raw_output = buffer.getvalue().strip()
        return self._clean_output(raw_output)

    @staticmethod
    def _clean_output(raw: str) -> str:
        """Remove DeepSeek-OCR-2's internal header/log lines."""
        if not raw:
            return "Error: Stream capture returned empty output."

        if "=====================" in raw:
            parts = raw.split("=====================")
            raw = parts[-1].strip()

        return raw or "Error: No content found after cleaning the model output."
