"""
engines/rag/retriever.py — Relevance Search & Retrieval for RAG Pipelines.

This module provides retrieval capabilities for Question-Answering (RAG) tasks.
Given a user natural language query and a collection of text chunks (with page metadata),
it scores, ranks, and retrieves the top-K most relevant chunks.
"""

import math
import re
from collections import Counter
from typing import Any, Dict, List


def _tokenize(text: str) -> List[str]:
    """
    Normalizes text by converting to lowercase and stripping punctuation,
    returning a list of word tokens.

    Args:
        text (str): Input text string.

    Returns:
        List[str]: Cleaned list of token strings.
    """
    if not text:
        return []
    # Find all alphanumeric word tokens, ignoring punctuation
    return re.findall(r"\w+", text.lower())


def _compute_idf(chunks: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Computes Inverse Document Frequency (IDF) for all unique terms across all chunks.

    IDF measures how informative a word is (rare words get higher weights).
    Formula: idf(term) = log( (N + 1) / (doc_freq(term) + 1) ) + 1
    """
    num_docs = len(chunks)
    doc_freq: Counter[str] = Counter()

    for chunk in chunks:
        content = chunk.get("content", "")
        unique_tokens = set(_tokenize(content))
        for token in unique_tokens:
            doc_freq[token] += 1

    idf_scores: Dict[str, float] = {}
    for token, freq in doc_freq.items():
        idf_scores[token] = math.log((num_docs + 1) / (freq + 1)) + 1.0

    return idf_scores


def _score_chunk_tfidf(
    query_tokens: List[str],
    chunk_tokens: List[str],
    idf_scores: Dict[str, float]
) -> float:
    """
    Calculates TF-IDF relevance score between a query token list and a chunk's tokens.

    Higher scores indicate higher overlap of important query keywords in the chunk.
    Includes length normalization to prevent long chunks from unfairly biasing the score.
    """
    if not query_tokens or not chunk_tokens:
        return 0.0

    chunk_tf = Counter(chunk_tokens)
    chunk_len = len(chunk_tokens)

    query_tf = Counter(query_tokens)

    score = 0.0
    for token in query_tf:
        if token in chunk_tf:
            # Term Frequency in chunk (normalized by chunk length)
            tf = chunk_tf[token] / chunk_len
            # Term Weight = TF * IDF
            idf = idf_scores.get(token, 1.0)
            # Query weight factor (giving extra weight if user repeated a keyword)
            query_weight = query_tf[token]

            score += tf * idf * query_weight

    return score


def retrieve_relevant_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Searches and retrieves the top-k most relevant chunks for a user query.

    Scoring Process:
      1. Tokenizes and normalizes query and chunk texts.
      2. Computes IDF weights across the input corpus.
      3. Scores each chunk using TF-IDF term overlap.
      4. Ranks chunks by relevance score descending.
      5. Returns top_k chunks with intact original schema (page_number, content).

    Args:
        query (str): Natural language user question.
        chunks (List[Dict]): List of chunk dicts from engines/rag/chunker.py.
        top_k (int): Number of top relevant chunks to retrieve. Default is 3.

    Returns:
        List[Dict]: Top `top_k` chunk dicts sorted by relevance score.
    """
    if not query or not chunks or top_k <= 0:
        return []

    # Tokenize user query
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    # Compute Corpus-wide IDF scores for keyword weighting
    idf_scores = _compute_idf(chunks)

    scored_chunks = []
    for index, chunk in enumerate(chunks):
        content = chunk.get("content", "")
        chunk_tokens = _tokenize(content)

        # Calculate TF-IDF relevance score
        score = _score_chunk_tfidf(query_tokens, chunk_tokens, idf_scores)

        # Keep track of score and original index for stable sorting
        scored_chunks.append({
            "score": score,
            "index": index,
            "chunk": chunk
        })

    # Sort chunks primarily by relevance score (descending), secondarily by original order
    scored_chunks.sort(key=lambda item: item["score"], reverse=True)

    # Filter out chunks with 0.0 score if we have non-zero matches, or return top_k candidates
    # Extract top_k chunk objects while retaining original schema
    top_results = [item["chunk"] for item in scored_chunks[:top_k]]

    return top_results


if __name__ == "__main__":
    # Example usage for testing and verification
    sample_chunks = [
        {
            "page_number": 1,
            "content": "This document outlines ISO 9001 quality management standards and safety protocols."
        },
        {
            "page_number": 1,
            "content": "The supplier Acme Corp produces automotive parts in Germany."
        },
        {
            "page_number": 2,
            "content": "Certificate ISO-9001-2024 was issued on 2024-01-15 by TÜV SÜD with expiration 2027."
        },
        {
            "page_number": 2,
            "content": "Payment terms and invoicing details for Acme Corp orders."
        }
    ]

    test_query = "What is the ISO 9001 certificate issue date and authority?"

    results = retrieve_relevant_chunks(test_query, sample_chunks, top_k=2)

    import json
    print(f"Query: '{test_query}'\n")
    print("Retrieved Chunks:")
    print(json.dumps(results, indent=2))
