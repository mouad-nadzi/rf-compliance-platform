"""
core/rag — Retrieval-Augmented Generation (RAG) Pipeline Package.

Provides document-aware chunking, dense/sparse embeddings, retrieval,
intent routing, and citation-backed Q&A synthesis.
"""

from core.rag.chunker import chunk_for_qa
from core.rag.embeddings import get_embedding, get_embeddings_batch
from core.rag.retriever import retrieve_relevant_chunks
from core.rag.qa import answer_query_with_citations
from core.rag.unified_agent import QueryIntent, unified_tool_select
from core.rag.sql_engine import execute_metadata_query
from core.rag.hybrid_engine import retrieve_hybrid_context, execute_unstructured_query
from core.rag.orchestrator import answer_compliance_query, answer_compliance_query_stream

__all__ = [
    "chunk_for_qa",
    "get_embedding",
    "get_embeddings_batch",
    "retrieve_relevant_chunks",
    "answer_query_with_citations",
    "QueryIntent",
    "unified_tool_select",
    "execute_metadata_query",
    "retrieve_hybrid_context",
    "execute_unstructured_query",
    "answer_compliance_query",
    "answer_compliance_query_stream",
]
