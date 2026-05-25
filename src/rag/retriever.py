from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from src.models import EvidenceChunk, EvidenceItem, EvidenceMap, ParsedChunk


TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣.%]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "") if len(token) > 1]


class EvidenceRetriever:
    """ChromaDB-compatible evidence retriever with lexical fallback.

    The MVP keeps a single public interface regardless of whether ChromaDB is
    installed. Chroma can be wired in later without changing downstream models.
    """

    def __init__(self, chunks: list[ParsedChunk], persist_dir: str | None = None, use_chroma: bool = True):
        self.chunks = chunks
        self.persist_dir = persist_dir
        self._vectors = [Counter(tokenize(chunk.text)) for chunk in chunks]
        self._collection = None
        if use_chroma and persist_dir and chunks:
            self._collection = self._try_init_chroma(persist_dir)

    def search(self, query: str, top_k: int = 3) -> list[EvidenceChunk]:
        if self._collection is not None:
            chroma_results = self._search_chroma(query, top_k)
            if chroma_results:
                return chroma_results
        query_vec = Counter(tokenize(query))
        scored: list[tuple[float, ParsedChunk]] = []
        for vector, chunk in zip(self._vectors, self.chunks):
            score = self._cosine(query_vec, vector)
            score *= self._source_weight(chunk)
            score *= self._quality_weight(chunk)
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            EvidenceChunk(
                evidence_id=f"ev_{idx + 1}_{chunk.chunk_id}",
                chunk_id=chunk.chunk_id,
                source_file=chunk.source_file,
                doc_type=chunk.doc_type,
                page_or_slide=chunk.page_or_slide,
                section_title=chunk.section_title,
                text=self._preview_text(chunk.text),
                score=round(score, 4),
                metadata=chunk.metadata,
            )
            for idx, (score, chunk) in enumerate(scored[:top_k])
        ]

    def build_evidence_map(self, field_claims: dict[str, list[str]], top_k: int = 3) -> EvidenceMap:
        items: list[EvidenceItem] = []
        for field, claims in field_claims.items():
            for claim in claims:
                if not claim:
                    continue
                items.append(EvidenceItem(field=field, claim=claim, evidence_chunks=self.search(claim, top_k=top_k)))
        return EvidenceMap(items=items)

    @staticmethod
    def _cosine(a: Counter, b: Counter) -> float:
        if not a or not b:
            return 0.0
        common = set(a) & set(b)
        numerator = sum(a[token] * b[token] for token in common)
        denom_a = math.sqrt(sum(value * value for value in a.values()))
        denom_b = math.sqrt(sum(value * value for value in b.values()))
        if denom_a == 0 or denom_b == 0:
            return 0.0
        return numerator / (denom_a * denom_b)

    def _try_init_chroma(self, persist_dir: str):
        # The deterministic lexical retriever is currently better for Korean
        # finance docs than hash embeddings. Keep Chroma disabled unless a real
        # embedding model is introduced.
        return None
        try:
            import chromadb
        except Exception:
            return None
        try:
            client = chromadb.PersistentClient(path=persist_dir)
            collection_name = "kolon_industry_chunks"
            existing = {collection.name for collection in client.list_collections()}
            if collection_name in existing:
                client.delete_collection(collection_name)
            collection = client.create_collection(collection_name, metadata={"hnsw:space": "cosine"})
            collection.add(
                ids=[chunk.chunk_id for chunk in self.chunks],
                documents=[chunk.text for chunk in self.chunks],
                embeddings=[self._hash_embedding(chunk.text) for chunk in self.chunks],
                metadatas=[
                    {
                        "source_file": chunk.source_file,
                        "doc_type": chunk.doc_type,
                        "page_or_slide": chunk.page_or_slide or "",
                        "section_title": chunk.section_title or "",
                        "element_type": chunk.element_type.value,
                        **{f"meta_{k}": str(v) for k, v in chunk.metadata.items()},
                    }
                    for chunk in self.chunks
                ],
            )
            return collection
        except Exception:
            return None

    def _search_chroma(self, query: str, top_k: int) -> list[EvidenceChunk]:
        try:
            result = self._collection.query(query_embeddings=[self._hash_embedding(query)], n_results=min(top_k, len(self.chunks)))
        except Exception:
            return []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        evidence: list[EvidenceChunk] = []
        for idx, chunk_id in enumerate(ids):
            meta = metas[idx] or {}
            distance = distances[idx] if idx < len(distances) else 1.0
            evidence.append(
                EvidenceChunk(
                    evidence_id=f"ev_{idx + 1}_{chunk_id}",
                    chunk_id=chunk_id,
                    source_file=meta.get("source_file", ""),
                    doc_type=meta.get("doc_type", ""),
                    page_or_slide=meta.get("page_or_slide") or None,
                    section_title=meta.get("section_title") or None,
                    text=self._preview_text(docs[idx]),
                    score=round(max(0.0, 1.0 - float(distance)), 4),
                    metadata=meta,
                )
            )
        return evidence

    @staticmethod
    def _hash_embedding(text: str, dims: int = 64) -> list[float]:
        vector = [0.0] * dims
        for token in tokenize(text):
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:2], "big") % dims
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    @staticmethod
    def _preview_text(text: str, max_chars: int = 520) -> str:
        text = re.sub(r"\s+", " ", text or "").strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "..."

    @staticmethod
    def _source_weight(chunk: ParsedChunk) -> float:
        name = chunk.source_file.lower()
        if chunk.doc_type in {"txt", "md"} or "pitch" in name or "녹취" in name:
            return 2.0
        if "memo" in name or "딜메모" in name or "투심" in name:
            return 1.8
        if chunk.doc_type == "docx":
            return 1.5
        if chunk.element_type.value == "table":
            return 0.45
        if chunk.doc_type == "pdf":
            return 0.85
        return 1.0

    @staticmethod
    def _quality_weight(chunk: ParsedChunk) -> float:
        text = chunk.text
        if not text:
            return 0.0
        weight = 1.0
        if 80 <= len(text) <= 900:
            weight += 0.25
        if any(keyword in text for keyword in ["시장", "고객", "경쟁", "리스크", "기회", "성장", "계약", "양산"]):
            weight += 0.3
        if text.count("|") >= 5 or text.count("•") >= 10:
            weight *= 0.45
        tokens = text.split()
        if tokens:
            numeric_like = sum(1 for token in tokens if re.search(r"\d", token))
            if numeric_like / len(tokens) > 0.35:
                weight *= 0.6
        return weight
