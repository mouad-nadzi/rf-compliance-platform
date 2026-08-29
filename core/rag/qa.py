"""
core/rag/qa.py — Q&A Synthesis & Citation Engine for RAG Pipeline.

Orchestrates Phase 3 Q&A answering: receives top relevant document chunks,
constructs structured LLM prompt payload, invokes Qwen GGUF LLM engine,
and returns a validated QAResponseSchema object with citations.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from core.prompts import QA_SYNTHESIS_SYSTEM_PROMPT
from schemas.qa import Citation, QAResponseSchema

logger = logging.getLogger(__name__)

FALLBACK_NOT_FOUND_MESSAGE = "Information not found in provided document context."

#: Matches the start of the JSON `"answer": "` field (value begins right after).
_ANSWER_FIELD_START_RE = re.compile(r'"answer"\s*:\s*"')


def _find_unescaped_quote(s: str) -> int:
    """Index of the first unescaped double-quote in a JSON string value, or -1."""
    escaped = False
    for i, ch in enumerate(s):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            return i
    return -1


def _trim_incomplete_escape(s: str) -> str:
    """Remove a trailing incomplete JSON escape (lone backslash or partial \\uXXXX)."""
    if s.endswith("\\"):
        return s[:-1]
    m = re.search(r"\\u([0-9a-fA-F]{0,3})$", s)
    if m and len(m.group(1)) < 4:
        return s[:m.start()]
    return s


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
        logger.error(f"  JSON/Pydantic validation failed for Q&A synthesis: {e}")
        logger.debug(f"Raw Output: {raw_json_response}")
        
        # Return fallback QAResponseSchema to prevent system crash
        return QAResponseSchema(
            question=query,
            answer=FALLBACK_NOT_FOUND_MESSAGE,
            citations=[]
        )


def extract_answer_value(buffer: str) -> Optional[str]:
    """
    Incrementally extracts the decoded text of the JSON 'answer' field from a
    partial streamed buffer.

    Handles reasoning blocks (<think>...</think>), JSON schemas with 'question' or
    other keys before 'answer', and escaped characters in partial strings.
    """
    if not buffer:
        return None

    # Strip thinking block if present to isolate JSON payload
    clean_buf = buffer
    if "</think>" in clean_buf:
        clean_buf = clean_buf.split("</think>", 1)[1]
    elif "<think>" in clean_buf:
        # Still inside thinking block
        return None

    # Search for "answer": " anywhere in the clean buffer
    m = _ANSWER_FIELD_START_RE.search(clean_buf)
    if m:
        raw = clean_buf[m.end():]
        end = _find_unescaped_quote(raw)
        if end != -1:
            raw = raw[:end]
        cleaned = _trim_incomplete_escape(raw)
        cleaned_escaped = (
            cleaned.replace("\r", "\\r")
            .replace("\n", "\\n")
            .replace("\t", "\\t")
        )
        try:
            return json.loads('"' + cleaned_escaped + '"')
        except Exception:
            return cleaned.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")

    # If the response is plain text (does not start with '{' or '```')
    s_buf = clean_buf.strip()
    if s_buf and not s_buf.startswith("{") and not s_buf.startswith("```"):
        if not s_buf.startswith('"') and not s_buf.startswith("</think"):
            return s_buf

    return None


def parse_qa_response(buffer: str, query: str) -> tuple:
    """
    Final-parse of a complete streamed buffer into (answer_text, citations_list).

    Uses the strict JSON ladder (extract_json) so reasoning traces and trailing
    tokens are scrubbed before QAResponseSchema validation.
    """
    from core.base import extract_json

    if not buffer or not buffer.strip():
        return "", []
    try:
        raw_json = extract_json(buffer)
        parsed = QAResponseSchema.model_validate_json(raw_json)
        answer = str(getattr(parsed, "answer", "") or "").strip()
        citations = [c.model_dump() for c in (parsed.citations or [])]
        return answer, citations
    except Exception as exc:
        logger.error(f"  Final Q&A parse failed for streamed buffer: {exc}")
        return "", []


def stream_synthesize_answer(
    system_prompt: str,
    user_prompt: str,
    query: str,
    disable_thinking: bool = False,
):
    """
    Streams a citation-backed answer generation token-by-token.

    Surfaces progressive text token-by-token as the JSON 'answer' field populates,
    while yielding raw thinking tokens separately during the reasoning phase.

    Args:
        system_prompt: System instructions for the completion.
        user_prompt:   User-facing query / instruction payload.
        query:         Original user question (used for final QAResponseSchema validation).
        disable_thinking: True for fast non-thinking mode (e.g. casual replies).

    Yields event dicts:
      {"type": "thinking", "text": <raw reasoning tokens>}
      {"type": "token", "text": <incremental answer text>}
      {"type": "synthesis_done", "answer": ..., "citations": [...]}
    """
    from core.llm import generate_stream

    buffer = ""
    last_answer = ""
    in_thinking_block = False

    for token in generate_stream(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        disable_thinking=disable_thinking,
    ):
        buffer += token

        # Handle live thinking events
        if "<think>" in token or (in_thinking_block and "</think>" not in token):
            in_thinking_block = True
            if not disable_thinking:
                clean_think = token.replace("<think>", "").replace("</think>", "")
                if clean_think:
                    yield {"type": "thinking", "text": clean_think}
            continue
        elif "</think>" in token:
            in_thinking_block = False
            if not disable_thinking:
                clean_think = token.split("</think>", 1)[0].replace("<think>", "")
                if clean_think:
                    yield {"type": "thinking", "text": clean_think}

        # Extract progressive answer text
        current = extract_answer_value(buffer)
        if current and len(current) > len(last_answer):
            delta = current[len(last_answer):]
            last_answer = current
            if delta:
                yield {"type": "token", "text": delta}

    answer, citations = parse_qa_response(buffer, query)
    final_answer = answer or last_answer
    
    if not last_answer and final_answer:
        yield {"type": "token", "text": final_answer}

    yield {
        "type": "synthesis_done",
        "answer": final_answer,
        "citations": citations,
    }


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
