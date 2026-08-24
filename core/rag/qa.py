"""
core/rag/qa.py — Q&A Synthesis & Citation Engine for RAG Pipeline.

Orchestrates Phase 3 Q&A answering: receives top relevant document chunks,
constructs structured LLM prompt payload, invokes Qwen GGUF LLM engine,
and returns a validated QAResponseSchema object with citations.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from core.prompts import QA_SYNTHESIS_SYSTEM_PROMPT
from schemas.qa import Citation, QAResponseSchema

logger = logging.getLogger(__name__)

FALLBACK_NOT_FOUND_MESSAGE = "Information not found in provided document context."


def _build_qa_prompt(query: str, retrieved_chunks: List[Dict[str, Any]]) -> tuple:
    """
    Constructs the structured prompt payload combining system instructions,
    retrieved chunk context blocks with metadata, and the user query.

    Returns:
        Tuple of (system_prompt, user_prompt) — the engine formats both into
        its own native chat template.
    """
    user_prompt = "--- BEGIN RETRIEVED DOCUMENT CONTEXT ---\n"

    for idx, chunk in enumerate(retrieved_chunks, start=1):
        file_name = chunk.get("file_name", "Unknown File")
        page_number = chunk.get("page_number", "Unknown Page")
        content = chunk.get("content", "").strip()

        user_prompt += f"\n[Context Chunk {idx}]\n"
        user_prompt += f"Source File: {file_name}\n"
        user_prompt += f"Page Number: {page_number}\n"
        user_prompt += f"Content:\n{content}\n"

    user_prompt += "\n--- END RETRIEVED DOCUMENT CONTEXT ---\n\n"
    user_prompt += f"USER QUESTION: {query}\n\n"
    user_prompt += "Return ONLY the raw JSON payload following the QAResponseSchema."

    return QA_SYNTHESIS_SYSTEM_PROMPT, user_prompt


def answer_query_with_citations(
    query: str,
    retrieved_chunks: List[Dict[str, Any]]
) -> QAResponseSchema:
    """
    Synthesizes a citation-backed answer for a user query given retrieved context chunks.

    Args:
        query (str): The user's natural language question.
        retrieved_chunks (List[Dict[str, Any]]): Top relevant chunk dictionaries produced
            by core/rag/retriever.py.

    Returns:
        QAResponseSchema: Validated Pydantic object containing question, synthesized answer,
            and supporting citations.
    """
    # Defensive check for empty inputs or empty chunks
    if not query or not query.strip() or not retrieved_chunks:
        logger.warning("Empty query or no retrieved chunks provided. Returning fallback response.")
        return QAResponseSchema(
            question=query if query else "",
            answer=FALLBACK_NOT_FOUND_MESSAGE,
            citations=[]
        )

    # 1. Build prompt payload combining context chunks and query
    system_prompt, user_prompt = _build_qa_prompt(query, retrieved_chunks)

    # 2. Invoke local LLM engine for JSON generation (Full Reasoning Mode)
    from core.llm import generate_json
    raw_json_response = generate_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        disable_thinking=False,
    )

    # 3. Parse and Validate with Pydantic schema
    try:
        parsed_response = QAResponseSchema.model_validate_json(raw_json_response)
        
        # Ensure question field matches original query if LLM altered it
        if not getattr(parsed_response, "question", None):
            parsed_response.question = query

        return parsed_response

    except Exception as e:
        logger.error(f"⚠️  JSON/Pydantic validation failed for Q&A synthesis: {e}")
        logger.debug(f"Raw Output: {raw_json_response}")
        
        # Return fallback QAResponseSchema to prevent system crash
        return QAResponseSchema(
            question=query,
            answer=FALLBACK_NOT_FOUND_MESSAGE,
            citations=[]
        )


if __name__ == "__main__":
    # Unit test demonstrating prompt construction and fallback behavior
    sample_chunks = [
        {
            "file_name": "supplier_packet.pdf",
            "document_id": "supplier_packet.pdf",
            "page_number": 1,
            "content": "Acme Corp is certified under ISO 9001 for quality management systems."
        },
        {
            "file_name": "supplier_packet.pdf",
            "document_id": "supplier_packet.pdf",
            "page_number": 2,
            "content": "Certificate number ISO-9001-2024 was issued on 2024-01-15 by TÜV SÜD."
        }
    ]

    test_query = "What is the certificate number and issuing authority?"
    print(f"Testing Q&A Prompt Construction for query: '{test_query}'\n")
    system_prompt, user_prompt = _build_qa_prompt(test_query, sample_chunks)
    print("Generated System Prompt:\n" + "-"*40)
    print(system_prompt)
    print("-"*40)
    print("Generated User Prompt:\n" + "-"*40)
    print(user_prompt)
    print("-"*40)

    # Test fallback validation
    fallback_res = answer_query_with_citations("What is the warranty policy?", [])
    print("\nFallback Response Output (Empty Chunks):")
    if hasattr(fallback_res, "model_dump"):
        print(json.dumps(fallback_res.model_dump(), indent=2))
    else:
        print(fallback_res.__dict__)
