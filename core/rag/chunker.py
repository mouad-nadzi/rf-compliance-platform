"""
core/rag/chunker.py — RAG Text Chunker for Q&A Pipelines.

This module provides document-aware and page-aware text chunking capabilities
for Question-Answering (RAG) tasks.
It parses layout-aware Markdown text containing page tags (e.g., '<Page 1>')
and splits it into paragraph chunks while tracking both page transitions and document identity.
"""

import re
from typing import Any, Dict, List, Optional, Union


# Regex pattern to match page delimiters like <Page 1>, <Page 2>, <Page AppendixA>, etc.
PAGE_TAG_PATTERN = re.compile(r"<Page\s+([^>]+)>", re.IGNORECASE)


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
        cleaned_paragraph = paragraph.strip()

        # Skip empty paragraphs resulting from multiple consecutive newlines
        if not cleaned_paragraph:
            continue

        # 3. Tag Detection: Check if this paragraph block contains a page tag (e.g., <Page 2>)
        match = PAGE_TAG_PATTERN.search(cleaned_paragraph)
        if match:
            raw_page_val = match.group(1).strip()

            # State Tracking Update: Convert to integer if numeric (e.g., "2" -> 2),
            # otherwise keep as string (e.g., "Appendix-A")
            if raw_page_val.isdigit():
                current_page = int(raw_page_val)
            else:
                current_page = raw_page_val

            # Remove the page tag from the paragraph text so tags don't pollute content
            cleaned_paragraph = PAGE_TAG_PATTERN.sub("", cleaned_paragraph).strip()

        # 4. Structuring: If valid content remains after tag removal, append to results list
        if cleaned_paragraph:
            chunk = {
                "file_name": file_name,
                "document_id": doc_id,
                "page_number": current_page,
                "content": cleaned_paragraph
            }
            chunks.append(chunk)

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
