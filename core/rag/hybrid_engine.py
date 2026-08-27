"""
core/rag/hybrid_engine.py — Hybrid Dense/Sparse RRF Engine with Parent Expansion.

Implements the unstructured RAG retrieval and Q&A synthesis pipeline:
  1. Dense Retrieval: Computes 1024-d query embeddings (BAAI/bge-m3) and performs pgvector search.
  2. Sparse Retrieval: Executes PostgreSQL full-text search & ILIKE keyword matching.
  3. Reciprocal Rank Fusion (RRF): Merges dense and sparse ranks with k=60.
  4. Adaptive Parent-Document Expansion: Expands to full document markdown when unique certs <= 3,
     otherwise falls back to top individual paragraph chunks.
  5. LLM Synthesis: Generates citation-backed answers via local LLM engine.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Tuple, Set

from sqlalchemy import func, or_, text
from sqlalchemy.exc import SQLAlchemyError

from server import config
from schemas.extraction import CertificateMetadata, CertificateChunk
from core.rag.embeddings import get_embedding

logger = logging.getLogger(__name__)

#: RRF smoothing constant
RRF_K: int = 60

#: Threshold for parent-document expansion vs chunk fallback
PARENT_EXPANSION_THRESHOLD: int = 3

#: Dense relevance gate: if even the best-matching chunk in the corpus has a
#: cosine distance above this value, the query has no meaningful match.
DENSE_RELEVANCE_THRESHOLD: float = 1.1

#: Fallback message when document context is unavailable
FALLBACK_NOT_FOUND_MESSAGE: str = "Information not found in provided document context."

#: English stopwords excluded from sparse tokenization (avoid broad ILIKE/FTS noise).
STOPWORDS: Set[str] = {
    "what", "which", "who", "where", "when", "why", "how", "about", "others",
    "other", "the", "a", "an", "and", "or", "but", "for", "with", "from",
    "are", "is", "was", "were", "be", "been", "being", "has", "have", "had",
    "do", "does", "did", "will", "would", "shall", "should", "can", "could",
    "may", "might", "must", "of", "to", "in", "on", "at", "by", "as", "it",
    "its", "them", "their", "they", "we", "you", "your", "this", "that",
    "these", "those", "there", "here", "not", "no", "so", "if", "then",
    "than", "too", "very", "just", "please", "tell", "list", "show", "give",
}


def _tokenize_query(query: str) -> List[str]:
    """Extracts clean alphanumeric tokens from query string (min length 2), excluding stopwords."""
    if not query:
        return []
    return [
        w for w in re.findall(r"\w+", query.lower())
        if len(w) >= 2 and w not in STOPWORDS
    ]


def _retrieve_dense_chunks(
    query_vector: List[float],
    db_session,
    candidate_limit: int = 20,
) -> List[CertificateChunk]:
    """
    Performs cosine similarity search using pgvector on certificate_chunks.embedding.
    """
    if not query_vector or all(v == 0.0 for v in query_vector):
        return []

    try:
        # Distance calculation using pgvector cosine_distance
        dense_results = (
            db_session.query(CertificateChunk)
            .filter(CertificateChunk.embedding.isnot(None))
            .order_by(CertificateChunk.embedding.cosine_distance(query_vector))
            .limit(candidate_limit)
            .all()
        )
        return dense_results
    except SQLAlchemyError as exc:
        logger.warning(f"  pgvector cosine distance query failed: {exc}. Rolling back transaction.")
        try:
            db_session.rollback()
        except Exception:
            pass
        return []
    except Exception as exc:
        logger.warning(f"  Unexpected error during dense vector retrieval: {exc}")
        try:
            db_session.rollback()
        except Exception:
            pass
        return []


def _retrieve_sparse_chunks(
    user_query: str,
    db_session,
    candidate_limit: int = 20,
) -> List[CertificateChunk]:
    """
    Performs lexical matching against certificate_chunks.raw_text using PostgreSQL FTS
    and ILIKE token search for part numbers, ISO standards, and keyword matches.
    """
    clean_query = str(user_query or "").strip()
    if not clean_query:
        return []

    tokens = _tokenize_query(clean_query)
    results_by_id: Dict[int, CertificateChunk] = {}
    ordered_chunks: List[CertificateChunk] = []

    # 1. PostgreSQL Full-Text Search (plainto_tsquery)
    try:
        ts_query = func.plainto_tsquery("english", clean_query)
        fts_chunks = (
            db_session.query(CertificateChunk)
            .filter(func.to_tsvector("english", CertificateChunk.raw_text).op("@@")(ts_query))
            .order_by(func.ts_rank_cd(func.to_tsvector("english", CertificateChunk.raw_text), ts_query).desc())
            .limit(candidate_limit)
            .all()
        )
        for chunk in fts_chunks:
            if chunk.id not in results_by_id:
                results_by_id[chunk.id] = chunk
                ordered_chunks.append(chunk)
    except Exception as exc:
        logger.debug(f"PostgreSQL FTS query skipped: {exc}")
        try:
            db_session.rollback()
        except Exception:
            pass

    # 2. ILIKE Token Search Fallback for exact codes, ISO standards, part numbers
    if len(ordered_chunks) < candidate_limit and tokens:
        try:
            ilike_filters = [CertificateChunk.raw_text.ilike(f"%{tok}%") for tok in tokens[:5]]
            ilike_chunks = (
                db_session.query(CertificateChunk)
                .filter(or_(*ilike_filters))
                .limit(candidate_limit)
                .all()
            )
            for chunk in ilike_chunks:
                if chunk.id not in results_by_id:
                    results_by_id[chunk.id] = chunk
                    ordered_chunks.append(chunk)
        except Exception as exc:
            logger.debug(f"ILIKE token search skipped: {exc}")
            try:
                db_session.rollback()
            except Exception:
                pass

    return ordered_chunks[:candidate_limit]


def compute_rrf_rankings(
    dense_chunks: List[CertificateChunk],
    sparse_chunks: List[CertificateChunk],
    k: int = RRF_K,
) -> List[Tuple[CertificateChunk, float]]:
    """
    Merges dense and sparse chunk rankings using Reciprocal Rank Fusion (RRF).

    RRF_Score(d) = sum( 1.0 / (k + rank(d)) ) for rank in {dense_rank, sparse_rank}

    Returns:
        List of (CertificateChunk, rrf_score) tuples sorted by rrf_score descending.
    """
    scores: Dict[int, float] = {}
    chunk_map: Dict[int, CertificateChunk] = {}

    # Rank dense results
    for rank_idx, chunk in enumerate(dense_chunks, start=1):
        c_id = chunk.id
        chunk_map[c_id] = chunk
        scores[c_id] = scores.get(c_id, 0.0) + (1.0 / (k + rank_idx))

    # Rank sparse results
    for rank_idx, chunk in enumerate(sparse_chunks, start=1):
        c_id = chunk.id
        chunk_map[c_id] = chunk
        scores[c_id] = scores.get(c_id, 0.0) + (1.0 / (k + rank_idx))

    # Sort candidates by combined RRF score descending
    sorted_candidates = sorted(
        [(chunk_map[c_id], rrf_score) for c_id, rrf_score in scores.items()],
        key=lambda pair: pair[1],
        reverse=True,
    )

    return sorted_candidates


def _dense_similarity_top(
    query_vector: List[float],
    db_session,
) -> Optional[float]:
    """Returns the best (lowest) pgvector cosine distance in the corpus for a query vector."""
    if not query_vector or all(v == 0.0 for v in query_vector):
        return None
    try:
        row = (
            db_session.query(
                CertificateChunk.embedding.cosine_distance(query_vector)
            )
            .filter(CertificateChunk.embedding.isnot(None))
            .order_by(CertificateChunk.embedding.cosine_distance(query_vector))
            .first()
        )
        return float(row[0]) if row is not None else None
    except Exception as exc:
        logger.debug(f"  Dense similarity probe failed: {exc}")
        return None


def _has_query_signal(query: str) -> bool:
    """A query with no meaningful tokens (stopwords/empty) carries no retrieval signal."""
    return len(_tokenize_query(query)) > 0


def retrieve_hybrid_context(
    user_query: str,
    db_session,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Executes hybrid retrieval (Dense + Sparse + RRF) with Adaptive Parent Expansion.

    Args:
        user_query: Natural language question asked by user.
        db_session: SQLAlchemy database session.
        top_k: Maximum candidate chunks to rank via RRF (default: 5).

    Returns:
        Dict payload containing:
          - retrieval_mode: "PARENT_EXPANSION", "CHUNK_FALLBACK", or "LOW_SIGNAL_QUERY"
          - context_text: Formatted markdown string preserving <Page X> tags
          - sources: List of source certificate dicts (file_name, certificate_id, pages)
          - top_chunks: List of top retrieved chunk objects
    """
    clean_query = str(user_query or "").strip()
    if not clean_query:
        return {
            "retrieval_mode": "CHUNK_FALLBACK",
            "context_text": "",
            "sources": [],
            "top_chunks": [],
        }

    # Relevance gate: a stopword-only / empty-token query has no retrieval signal.
    if not _has_query_signal(clean_query):
        logger.info(f" Query '{clean_query!r}' carries no retrieval signal (stopword-only). Skipping RAG retrieval.")
        return {
            "retrieval_mode": "LOW_SIGNAL_QUERY",
            "context_text": "",
            "sources": [],
            "top_chunks": [],
        }

    # 1. Compute Dense Query Vector
    query_vector = get_embedding(clean_query)

    # 2. Dual Retrieval (Dense + Sparse)
    candidate_limit = max(top_k * 4, 20)
    dense_chunks = _retrieve_dense_chunks(query_vector, db_session, candidate_limit=candidate_limit)

    # Dense relevance gate: reject when even the best corpus match is unrelated.
    best_dist = _dense_similarity_top(query_vector, db_session)
    if best_dist is not None and best_dist > DENSE_RELEVANCE_THRESHOLD:
        logger.info(f" Dense relevance gate rejected retrieval (best cosine distance {best_dist:.3f}).")
        dense_chunks = []

    sparse_chunks = _retrieve_sparse_chunks(clean_query, db_session, candidate_limit=candidate_limit)

    # 3. Reciprocal Rank Fusion (RRF)
    rrf_ranked = compute_rrf_rankings(dense_chunks, sparse_chunks, k=RRF_K)
    top_candidates = [pair[0] for pair in rrf_ranked[:top_k]]

    if not top_candidates:
        logger.warning(f"No candidate chunks retrieved for query: {clean_query!r}")
        return {
            "retrieval_mode": "CHUNK_FALLBACK",
            "context_text": "",
            "sources": [],
            "top_chunks": [],
        }

    # 4. Adaptive Parent-Document Expansion Decision
    unique_cert_ids = list(dict.fromkeys([c.certificate_id for c in top_candidates if c.certificate_id]))

    if len(unique_cert_ids) <= PARENT_EXPANSION_THRESHOLD:
        retrieval_mode = "PARENT_EXPANSION"
        logger.info(
            f" Parent-Document Expansion active ({len(unique_cert_ids)} unique certificates <= {PARENT_EXPANSION_THRESHOLD})"
        )

        context_blocks = []
        sources = []

        for cert_id in unique_cert_ids:
            # Query metadata
            cert = db_session.query(CertificateMetadata).filter(CertificateMetadata.certificate_id == cert_id).first()
            file_name = cert.file_name if cert and cert.file_name else cert_id
            supplier = cert.supplier if cert and cert.supplier else "Unknown"

            # Query ALL chunks for this certificate ordered by page_number, id
            all_chunks = (
                db_session.query(CertificateChunk)
                .filter(CertificateChunk.certificate_id == cert_id)
                .order_by(CertificateChunk.page_number.asc(), CertificateChunk.id.asc())
                .all()
            )

            pages_present = list(dict.fromkeys([chk.page_number for chk in all_chunks if chk.page_number]))
            full_doc_markdown = "\n\n".join([chk.raw_text for chk in all_chunks if chk.raw_text])

            context_blocks.append(
                f"=== DOCUMENT SOURCE: {file_name} (ID: {cert_id}) ===\n"
                f"Supplier: {supplier} | Pages Included: {pages_present}\n\n"
                f"{full_doc_markdown}"
            )

            sources.append({
                "certificate_id": cert_id,
                "file_name": file_name,
                "supplier": supplier,
                "pages": pages_present,
            })

        context_text = "\n\n".join(context_blocks)

    else:
        retrieval_mode = "CHUNK_FALLBACK"
        logger.info(
            f" Chunk Fallback active ({len(unique_cert_ids)} unique certificates > {PARENT_EXPANSION_THRESHOLD})"
        )

        context_blocks = []
        sources = []

        for idx, chk in enumerate(top_candidates, start=1):
            cert_id = chk.certificate_id
            cert = db_session.query(CertificateMetadata).filter(CertificateMetadata.certificate_id == cert_id).first()
            file_name = cert.file_name if cert and cert.file_name else cert_id
            supplier = cert.supplier if cert and cert.supplier else "Unknown"
            page_num = chk.page_number or 1

            context_blocks.append(
                f"=== CHUNK {idx} (File: {file_name}, Page: {page_num}, Cert ID: {cert_id}) ===\n"
                f"Supplier: {supplier}\n"
                f"{chk.raw_text}"
            )

            sources.append({
                "certificate_id": cert_id,
                "file_name": file_name,
                "supplier": supplier,
                "pages": [page_num],
            })

        context_text = "\n\n".join(context_blocks)

    return {
        "retrieval_mode": retrieval_mode,
        "context_text": context_text,
        "sources": sources,
        "top_chunks": top_candidates,
    }


def build_unstructured_qa_prompt(user_query: str, context_text: str, history_text: str = "") -> str:
    """Builds the QA synthesis user prompt from retrieved context (shared by sync + streaming paths)."""
    history_block = f"{history_text}\n\n" if history_text else ""
    return (
        f"{history_block}"
        f"--- BEGIN RETRIEVED DOCUMENT CONTEXT ---\n"
        f"{context_text}\n"
        f"--- END RETRIEVED DOCUMENT CONTEXT ---\n\n"
        f"USER QUESTION: {user_query}\n\n"
        f"Return ONLY the raw JSON output matching the QAResponseSchema schema."
    )


def execute_unstructured_query(user_query: str, db_session=None, history_text: str = "") -> str:
    """
    Full UNSTRUCTURED_RAG execution pipeline:
      1. Hybrid retrieval (Dense + Sparse + RRF + Parent Expansion).
      2. Formats QA prompt payload preserving <Page X> tags for citation generation.
      3. Synthesizes a natural-language answer via local LLM engine.

    Args:
        user_query (str): Natural language user question.
        db_session: SQLAlchemy session (or None to open/close local session).
        history_text (str): Optional prior conversation history block.

    Returns:
        str: Synthesized natural-language answer with inline citations.
    """
    clean_query = str(user_query or "").strip()
    if not clean_query:
        return FALLBACK_NOT_FOUND_MESSAGE

    close_session_on_exit = False
    if db_session is None:
        from storage.database import SessionLocal

        db_session = SessionLocal()
        close_session_on_exit = True

    try:
        # 1. Retrieve Hybrid Context
        retrieval_payload = retrieve_hybrid_context(clean_query, db_session, top_k=5)
        context_text = retrieval_payload.get("context_text", "").strip()

        if not context_text:
            return FALLBACK_NOT_FOUND_MESSAGE

        # 2. Build QA Synthesis Prompt Payload
        from core.prompts import QA_SYNTHESIS_SYSTEM_PROMPT
        user_prompt = build_unstructured_qa_prompt(clean_query, context_text, history_text)

        # 3. LLM Answer Synthesis
        from core.llm import generate_json
        raw_json_response = generate_json(
            system_prompt=QA_SYNTHESIS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            disable_thinking=False,
        )

        # 4. Parse JSON answer output
        try:
            import json

            parsed = json.loads(raw_json_response)
            if isinstance(parsed, dict) and "answer" in parsed:
                ans_text = str(parsed.get("answer", "")).strip()
                if ans_text:
                    return ans_text
        except Exception:
            pass

        clean_fallback = re.sub(r"```(?:json)?\s*", "", raw_json_response, flags=re.IGNORECASE).strip()
        clean_fallback = re.sub(r"\s*```$", "", clean_fallback).strip()

        return clean_fallback if clean_fallback else FALLBACK_NOT_FOUND_MESSAGE

    except Exception as exc:
        logger.warning(f"  Unstructured RAG execution failed: {exc}")
        return FALLBACK_NOT_FOUND_MESSAGE

    finally:
        if close_session_on_exit:
            db_session.close()


if __name__ == "__main__":
    print("Testing Hybrid Engine imports and structure...")
    print(f"RRF_K: {RRF_K}")
    print(f"PARENT_EXPANSION_THRESHOLD: {PARENT_EXPANSION_THRESHOLD}")
    print(" hybrid_engine.py module defined successfully.")
