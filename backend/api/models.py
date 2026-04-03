"""Pydantic models for the query API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class QueryRequest(BaseModel):
    """Public request contract for sync and streaming query endpoints."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1)
    doc_id: str | None = None
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    languages: list[str] = Field(default_factory=list)
    scripts: list[str] = Field(default_factory=list)
    chunk_types: list[str] = Field(default_factory=list)
    include_debug: bool = False

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        cleaned = " ".join(str(value or "").split()).strip()
        if not cleaned:
            raise ValueError("query cannot be empty")
        return cleaned

    @field_validator("languages", "scripts", "chunk_types")
    @classmethod
    def clean_string_lists(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values or []:
            item = str(value or "").strip()
            if item and item not in cleaned:
                cleaned.append(item)
        return cleaned

    @model_validator(mode="after")
    def validate_page_range(self):
        if self.page_start is not None and self.page_end is not None and self.page_end < self.page_start:
            raise ValueError("page_end must be greater than or equal to page_start")
        return self


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: str
    doc_id: str
    source_file: str
    page_numbers: list[int]
    section_path: list[str] = Field(default_factory=list)
    snippet: str


class ImageCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    image_id: str
    page_number: int | None = None
    caption: str = ""
    labels: list[str] = Field(default_factory=list)
    image_url: str
    cloudinary_public_id: str | None = None
    source_file: str
    citation_ids: list[str] = Field(default_factory=list)


class TableCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    table_id: str
    page_numbers: list[int]
    table_caption: str | None = None
    table_markdown: str | None = None
    table_rows: list[list[str]] | None = None
    source_file: str
    citation_ids: list[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    images: list[ImageCard] = Field(default_factory=list)
    tables: list[TableCard] = Field(default_factory=list)
    enough_evidence: bool
    query_intent: str
    model: str
    timings: dict[str, float] = Field(default_factory=dict)
    debug: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    llm_provider: str
    llm_available: bool
    qdrant_reachable: bool
    text_collection: str
    image_collection: str
