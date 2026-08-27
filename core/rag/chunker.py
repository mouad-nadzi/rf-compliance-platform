"""
core/rag/chunker.py — RAG Text Chunker for Q&A Pipelines.

This module provides document-aware and page-aware text chunking capabilities
for Question-Answering (RAG) tasks.
It parses layout-aware Markdown text containing page tags (e.g., '<Page 1>')
and splits it into paragraph chunks while tracking both page transitions and document identity.

Chunking strategy:
  - Primary boundary: double-newline (`\n\n`) paragraphs. No overlap (0 tokens).
  - Safety valve: OCR engines occasionally emit giant, unformatted text blocks
    (e.g. a complex table missing double newlines). If a single paragraph
    exceeds `SOFT_MAX_CHUNK_TOKENS` (~800 tokens), that paragraph alone is
    split on single newlines (`\n`) first, then on period boundaries (`. `),
    so no oversized chunk is ever indexed.
"""

import re
from typing import Any, Dict, List, Optional, Union


# Regex pattern to match page delimiters like <Page 1>, <Page 2>, <Page AppendixA>, etc.
PAGE_TAG_PATTERN = re.compile(r"<Page\s+([^>]+)>", re.IGNORECASE)

#: Soft cap (tokens) above which a single paragraph triggers the fallback splitter.
#: ~4 characters per token (consistent with the codebase token estimator).
SOFT_MAX_CHUNK_TOKENS: int = 800

#: Sentence-boundary regex for the final fallback split (periods followed by whitespace).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def estimate_tokens(text: str) -> int:
    """Rough token estimator (~4 chars per token), matching the rest of the codebase."""
    return max(1, len(str(text or "")) // 4)


def _pack_parts(parts: List[str]) -> List[str]:
    """
    Greedily packs split parts into chunks that stay at or below the soft cap.
    A single part larger than the cap (e.g. one long sentence) is kept whole.
    """
    chunks: List[str] = []
    current = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        candidate = f"{current} {part}".strip() if current else part
        if current and estimate_tokens(candidate) > SOFT_MAX_CHUNK_TOKENS:
            chunks.append(current)
            current = part
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _split_long_paragraph(paragraph: str) -> List[str]:
    """
    Splits a single oversized paragraph into sub-chunks under the soft cap.

    Level 1: split on single newlines (`\n`) — handles multi-row tables/blocks
             that OCR emitted without blank-line paragraph breaks.
    Level 2: split on period boundaries (`. `) — handles giant unbroken prose.
    """
    if estimate_tokens(paragraph) <= SOFT_MAX_CHUNK_TOKENS:
        return [paragraph]

    newline_parts = [p.strip() for p in re.split(r"\n+", paragraph) if p.strip()]
    if len(newline_parts) > 1:
        # Further split any individual part that is still oversized (a giant single
        # line), then pack all pieces into cap-sized chunks (no overlap).
        expanded: List[str] = []
        for part in newline_parts:
            if estimate_tokens(part) > SOFT_MAX_CHUNK_TOKENS:
                expanded.extend(_split_long_paragraph(part))
            else:
                expanded.append(part)
        return _pack_parts(expanded)

    sentence_parts = [s.strip() for s in _SENTENCE_SPLIT_RE.split(paragraph) if s.strip()]
    if len(sentence_parts) > 1:
        return _pack_parts(sentence_parts)

    return [paragraph]


def _process_block(block_text: str, current_page: Union[int, str]) -> tuple:
    """
    Cleans a text block and detects/strips any page tag.

    Returns:
        (cleaned_content, updated_page)
    """
    cleaned = block_text.strip()
    match = PAGE_TAG_PATTERN.search(cleaned)
    if match:
        raw_page_val = match.group(1).strip()
        if raw_page_val.isdigit():
            current_page = int(raw_page_val)
        else:
            current_page = raw_page_val
        cleaned = PAGE_TAG_PATTERN.sub("", cleaned).strip()
    return cleaned, current_page


def chunk_for_qa(
    markdown_text: str,
    file_name: str,
    document_id: Optional[str] = None
) -> List[Dict[str, Union[int, str]]]:
    """
    Splits layout-aware Markdown text into paragraph chunks with document and page metadata.

    State Tracking:
      - Tracks the current page number as it processes the document sequentially (default 1).
      - Attaches document identity metadata ('file_name' and 'document_id') to every chunk.

    Args:
        markdown_text (str): Raw Markdown string containing text and custom page tags.
        file_name (str): The source filename of the document (e.g., "supplier_packet.pdf").
        document_id (Optional[str]): Unique identifier for the document.
                                     Defaults to `file_name` if not provided.

    Returns:
        List[Dict[str, Union[int, str]]]: List of structured chunk dictionaries containing:
            - "file_name": Source document filename
            - "document_id": Unique document identifier
            - "page_number": Integer or string representing page number
            - "content": Clean paragraph text content
    """
    if not markdown_text or not isinstance(markdown_text, str):
        return []

    # Resolve document identity default
    doc_id = document_id if (document_id is not None and str(document_id).strip()) else file_name

    # 1. State Tracking: Initialize the page tracker to 1 (default page number)
    current_page: Union[int, str] = 1

    # 2. Paragraph Splitting: Divide text into blocks separated by double-newlines (\n\n)
    raw_paragraphs = markdown_text.split("\n\n")

    chunks: List[Dict[str, Union[int, str]]] = []

    for paragraph in raw_paragraphs:
        # Strip leading/trailing whitespace from the paragraph block
        cleaned_paragraph, current_page = _process_block(paragraph, current_page)

        # Skip empty paragraphs resulting from multiple consecutive newlines
        if not cleaned_paragraph:
            continue

        def _append(content: str) -> None:
            chunks.append({
                "file_name": file_name,
                "document_id": doc_id,
                "page_number": current_page,
                "content": content
            })

        # 3. Safety Valve: split oversized paragraphs (giant OCR blocks / tables)
        #    on single newlines then period boundaries, without adding overlap.
        if estimate_tokens(cleaned_paragraph) > SOFT_MAX_CHUNK_TOKENS:
            for sub in _split_long_paragraph(cleaned_paragraph):
                sub_clean, current_page = _process_block(sub, current_page)
                if sub_clean:
                    _append(sub_clean)
        else:
            _append(cleaned_paragraph)

    return chunks


if __name__ == "__main__":
    # Example usage for testing and verification
    sample_md = """
    <Page 1>

    This is the first paragraph of the document on page 1. It contains general information.

    Here is a second paragraph on the same page describing ISO standards.

    <Page 2>

    Now we have moved to page 2. This paragraph details supplier information.

    Final notes and sign-off section.
    """

    result = chunk_for_qa(
        markdown_text=sample_md,
        file_name="supplier_packet.pdf",
        document_id="doc_12345"
    )
    import json
    print(json.dumps(result, indent=2))
