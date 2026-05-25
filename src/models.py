from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ElementType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE_SUMMARY = "image_summary"
    SLIDE = "slide"
    TRANSCRIPT = "transcript"


class ParsedChunk(BaseModel):
    chunk_id: str
    source_file: str
    doc_type: str
    page_or_slide: str | None = None
    section_title: str | None = None
    text: str
    element_type: ElementType = ElementType.TEXT
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("ParsedChunk.text cannot be empty")
        return value.strip()


class EvidenceChunk(BaseModel):
    evidence_id: str
    chunk_id: str
    source_file: str
    doc_type: str
    page_or_slide: str | None = None
    section_title: str | None = None
    text: str
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceItem(BaseModel):
    field: str
    claim: str
    evidence_chunks: list[EvidenceChunk] = Field(default_factory=list)


class EvidenceMap(BaseModel):
    items: list[EvidenceItem] = Field(default_factory=list)

    def evidence_for_field(self, field: str) -> list[EvidenceItem]:
        return [item for item in self.items if item.field == field]

    def has_evidence(self, field: str) -> bool:
        return any(item.evidence_chunks for item in self.evidence_for_field(field))


class IndustryExtraction(BaseModel):
    company_name: str = ""
    industry: str = ""
    target_market: str = ""
    market_size: str = ""
    market_growth_rate: str = ""
    growth_drivers: list[str] = Field(default_factory=list)
    customer_pain: list[str] = Field(default_factory=list)
    competition: list[str] = Field(default_factory=list)
    regulation_or_policy: list[str] = Field(default_factory=list)
    opportunity: list[str] = Field(default_factory=list)
    risk: list[str] = Field(default_factory=list)


class KGNode(BaseModel):
    id: str
    type: str
    label: str
    properties: dict[str, Any] = Field(default_factory=dict)


class KGEdge(BaseModel):
    source: str
    relation: str
    target: str
    properties: dict[str, Any] = Field(default_factory=dict)


class MiniKG(BaseModel):
    nodes: list[KGNode] = Field(default_factory=list)
    edges: list[KGEdge] = Field(default_factory=list)


class ValidationWarning(BaseModel):
    rule: str
    severity: str
    message: str
    field: str | None = None


class ValidationReport(BaseModel):
    passed: list[str] = Field(default_factory=list)
    warnings: list[ValidationWarning] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(w.severity == "high" for w in self.warnings)
