"""
schemas/qa.py — Pydantic Schemas for RAG Q&A Synthesis & Citation.

Defines the structure for question answering responses and fine-grained
citations referencing original source file names, page numbers, and quotes.
"""

from typing import Any, Dict, List, Union

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    
    # Lightweight fallback model for local python environments without pydantic installed
    class BaseModel:
        def __init__(self, **data):
            for k, v in data.items():
                setattr(self, k, v)

        def model_dump(self) -> Dict[str, Any]:
            res = {}
            for k, v in self.__dict__.items():
                if isinstance(v, list):
                    res[k] = [item.model_dump() if hasattr(item, "model_dump") else item for item in v]
                elif hasattr(v, "model_dump"):
                    res[k] = v.model_dump()
                else:
                    res[k] = v
            return res

        @classmethod
        def model_validate_json(cls, json_str: str):
            import json
            data = json.loads(json_str)
            return cls.model_validate(data)

        @classmethod
        def model_validate(cls, data: Dict[str, Any]):
            if "citations" in data and isinstance(data["citations"], list):
                validated_citations = []
                for c in data["citations"]:
                    if isinstance(c, dict):
                        validated_citations.append(Citation(**c))
                    else:
                        validated_citations.append(c)
                data["citations"] = validated_citations
            return cls(**data)

    def Field(default=None, default_factory=None, description=""):
        if default_factory is not None:
            return default_factory()
        return default


if PYDANTIC_AVAILABLE:
    class Citation(BaseModel):
        file_name: str = Field(
            ...,
            description="The source filename where the supporting quote is located."
        )
        page_number: Union[int, str] = Field(
            ...,
            description="The page number (integer or string) where the supporting quote was found."
        )
        supporting_quote: str = Field(
            ...,
            description="The exact quote or key sentence from the document chunk that supports the answer."
        )

    class QAResponseSchema(BaseModel):
        question: str = Field(
            ...,
            description="The original user query or question."
        )
        answer: str = Field(
            ...,
            description="The synthesized answer based strictly on the retrieved document context."
        )
        citations: List[Citation] = Field(
            default_factory=list,
            description="List of supporting citations with document source metadata and quotes."
        )
else:
    class Citation(BaseModel):
        file_name: str
        page_number: Union[int, str]
        supporting_quote: str

    class QAResponseSchema(BaseModel):
        question: str
        answer: str
        citations: List[Citation] = []
