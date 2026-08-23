"""
engines/rag — Retrieval-Augmented Generation (RAG) Pipeline Package.

Provides document-aware chunking, dense/sparse embeddings, retrieval,
intent routing, and citation-backed Q&A synthesis.
"""

from engines.rag.chunker import chunk_for_qa
from engines.rag.embeddings import get_embedding, get_embeddings_batch
from engines.rag.retriever import retrieve_relevant_chunks
from engines.rag.qa import answer_query_with_citations
from engines.rag.router import QueryIntent, RouterDecision, classify_intent
from engines.rag.sql_engine import execute_metadata_query
from engines.rag.hybrid_engine import retrieve_hybrid_context, execute_unstructured_query
from engines.rag.orchestrator import answer_compliance_query

__all__ = [
    "chunk_for_qa",
    "get_embedding",
    "get_embeddings_batch",
    "retrieve_relevant_chunks",
    "answer_query_with_citations",
    "QueryIntent",
    "RouterDecision",
    "classify_intent",
    "execute_metadata_query",
    "retrieve_hybrid_context",
    "execute_unstructured_query",
    "answer_compliance_query",
]
