"""
core/rag/embeddings.py — Unified Embedding Generation Facade (BAAI/bge-m3)

Provides 1024-dimensional dense vector embeddings using the BAAI/bge-m3 model.
Supports single text encoding and batch encoding with fallback handling.
"""

import logging
from typing import List, Optional

from server import config

logger = logging.getLogger(__name__)

_embedding_model = None


def get_embedding_model():
    """
    Lazy loads the BAAI/bge-m3 SentenceTransformer model into memory.
    """
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            model_name = getattr(config, "EMBEDDING_MODEL", "BAAI/bge-m3")
            cache_dir = getattr(config, "CACHE_DIR", None)
            device = getattr(config, "EMBEDDING_DEVICE", "cpu")
            logger.info(f"Loading embedding model '{model_name}' on '{device}'...")
            # bge-m3 runs on CPU by default: embeddings are per-chunk during
            # ingestion, so CPU latency (~2-4 s/batch) is acceptable, and it keeps
            # the T4's ~15 GB entirely for Gemma + OCR inference headroom.
            _embedding_model = SentenceTransformer(
                model_name,
                cache_folder=cache_dir,
                device=device,
            )
            logger.info(f"✅ Embedding model '{model_name}' loaded successfully on '{device}'.")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load SentenceTransformer embedding model: {e}")
            _embedding_model = False  # Mark load attempt failed
    return _embedding_model if _embedding_model is not False else None


def get_embedding(text: str) -> List[float]:
    """
    Computes a 1024-dimensional dense vector embedding for a single text input.

    Args:
        text (str): Input text string.

    Returns:
        List[float]: 1024-dimensional embedding vector.
    """
    if not text or not text.strip():
        return [0.0] * getattr(config, "EMBEDDING_DIM", 1024)

    model = get_embedding_model()
    if model is not None:
        try:
            embedding = model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"❌ Error computing embedding: {e}")

    # Fallback: 1024 zero-vector if model is unavailable
    return [0.0] * getattr(config, "EMBEDDING_DIM", 1024)


def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Computes 1024-dimensional dense vector embeddings for a list of text inputs.

    Args:
        texts (List[str]): List of input text strings.

    Returns:
        List[List[float]]: List of 1024-dimensional embedding vectors.
    """
    if not texts:
        return []

    model = get_embedding_model()
    if model is not None:
        try:
            batch_size = getattr(config, "EMBEDDING_BATCH_SIZE", 16)
            embeddings = model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=batch_size,
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"❌ Error computing batch embeddings: {e}")

    # Fallback zero vectors for batch
    dim = getattr(config, "EMBEDDING_DIM", 1024)
    return [[0.0] * dim for _ in texts]
