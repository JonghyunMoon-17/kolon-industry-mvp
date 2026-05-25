from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any

from src.llm import BaseLLMProvider
from src.models import IndustryExtraction, ParsedChunk
from src.pipeline_state import PipelineDiagnostics


FIELD_KEYWORDS = {
    "industry": ["산업", "sector", "industry"],
    "target_market": ["시장", "market", "tam", "sam", "som"],
    "market_size": ["시장규모", "규모", "조원", "억원", "billion", "million", "tam"],
    "market_growth_rate": ["성장률", "cagr", "%", "연평균"],
    "growth_drivers": ["성장", "확대", "수요", "트렌드", "driver", "growth"],
    "customer_pain": ["문제", "pain", "니즈", "불편", "비용", "고객"],
    "competition": ["경쟁", "competitor", "player", "플레이어", "대체재"],
    "regulation_or_policy": ["규제", "정책", "법", "정부", "인허가"],
    "opportunity": ["기회", "opportunity", "확장", "진입", "수혜"],
    "risk": ["리스크", "위험", "검증", "불확실", "risk", "한계"],
}

NOISE_HINTS = [
    "confidential",
    "chip photo",
    "package",
    "scanning",
    "resolution",
    "status:",
    "m.p:",
    "seed",
    "series-",
]

COMPANY_PATTERNS = [
    re.compile(r"([A-Za-z0-9가-힣][A-Za-z0-9가-힣\s\-\(\)]{1,30})(?:은|는)\s"),
    re.compile(r"회사명\s*[:：]\s*([^\n]+)"),
    re.compile(r"기업명\s*[:：]\s*([^\n]+)"),
]


class IndustryExtractor:
    """Schema-bound extraction.

    Real LLM providers can replace `_heuristic_extract`; the public return type
    stays fixed as `IndustryExtraction`.
    """

    def extract(
        self,
        chunks: list[ParsedChunk],
        provider: BaseLLMProvider | None = None,
        diagnostics: PipelineDiagnostics | None = None,
    ) -> IndustryExtraction:
        if provider is not None and chunks:
            try:
                result = self._llm_extract(chunks, provider)
                if diagnostics:
                    diagnostics.add("Claude extraction succeeded")
                return result
            except Exception as exc:
                if diagnostics:
                    diagnostics.add(f"Claude extraction failed; heuristic fallback used: {exc}")
        return self._heuristic_extract(chunks)

    def _llm_extract(self, chunks: list[ParsedChunk], provider: BaseLLMProvider) -> IndustryExtraction:
        prompt = self._build_extraction_prompt(chunks)
        data = provider.generate_structured(prompt, IndustryExtraction.model_json_schema())
        return IndustryExtraction.model_validate(data)

    def _heuristic_extract(self, chunks: list[ParsedChunk]) -> IndustryExtraction:
        text = "\n".join(chunk.text for chunk in chunks)
        extraction = IndustryExtraction(
            company_name=self._guess_company(text),
            industry=self._first_sentence_by_keywords(chunks, FIELD_KEYWORDS["industry"]),
            target_market=self._first_sentence_by_keywords(chunks, FIELD_KEYWORDS["target_market"]),
            market_size=self._first_sentence_by_keywords(chunks, FIELD_KEYWORDS["market_size"], require_number=True),
            market_growth_rate=self._first_sentence_by_keywords(chunks, FIELD_KEYWORDS["market_growth_rate"], require_number=True),
            growth_drivers=self._top_sentences(chunks, FIELD_KEYWORDS["growth_drivers"], limit=3),
            customer_pain=self._top_sentences(chunks, FIELD_KEYWORDS["customer_pain"], limit=3),
            competition=self._top_sentences(chunks, FIELD_KEYWORDS["competition"], limit=3),
            regulation_or_policy=self._top_sentences(chunks, FIELD_KEYWORDS["regulation_or_policy"], limit=2),
            opportunity=self._top_sentences(chunks, FIELD_KEYWORDS["opportunity"], limit=3),
            risk=self._top_sentences(chunks, FIELD_KEYWORDS["risk"], limit=3),
        )
        return IndustryExtraction.model_validate(extraction.model_dump())

    @staticmethod
    def _build_extraction_prompt(chunks: list[ParsedChunk], max_chars: int = 18000) -> str:
        excerpts: list[str] = []
        used = 0
        for chunk in IndustryExtractor._select_extraction_chunks(chunks):
            text = chunk.text[:1200]
            header = f"[{chunk.chunk_id}] file={chunk.source_file}, type={chunk.doc_type}, page_or_slide={chunk.page_or_slide or '-'}"
            block = f"{header}\n{text}"
            if used + len(block) > max_chars:
                break
            excerpts.append(block)
            used += len(block)
        return (
            "당신은 VC 투자심사보고서의 '산업 현황 및 분석' 섹션을 위한 정보 추출 에이전트입니다.\n"
            "아래 입력자료에서 산업분석에 필요한 정보만 추출해 JSON schema에 맞춰 반환하세요.\n\n"
            "규칙:\n"
            "- 입력 문서에 없는 사실은 만들지 마세요.\n"
            "- 제품 스펙, 표의 숫자, 회사 홍보 문구를 그대로 길게 복사하지 마세요.\n"
            "- 각 필드는 짧고 보고서에 쓰기 좋은 문장 또는 구문으로 정리하세요.\n"
            "- market_size와 market_growth_rate는 수치 근거가 보일 때만 채우세요.\n"
            "- opportunity와 risk는 투자대상기업과 연결되는 산업/시장 관점으로 작성하세요.\n\n"
            "<documents>\n"
            + "\n\n---\n\n".join(excerpts)
            + "\n</documents>"
        )

    @staticmethod
    def _select_extraction_chunks(chunks: list[ParsedChunk], limit: int = 24) -> list[ParsedChunk]:
        query_keywords = [
            "시장",
            "산업",
            "성장",
            "수요",
            "고객",
            "문제",
            "경쟁",
            "리스크",
            "기회",
            "매출",
            "양산",
            "계약",
            "투자",
        ]

        def source_priority(chunk: ParsedChunk) -> int:
            name = chunk.source_file.lower()
            if chunk.doc_type in {"txt", "md"} or "pitch" in name or "녹취" in name:
                return 5
            if "memo" in name or "딜메모" in name or "투심" in name:
                return 4
            if chunk.doc_type == "docx":
                return 3
            if chunk.doc_type == "pptx":
                return 2
            return 1

        def score(chunk: ParsedChunk) -> tuple[int, int, int]:
            text = chunk.text
            keyword_score = sum(text.count(keyword) for keyword in query_keywords)
            length_score = 1 if 80 <= len(text) <= 1200 else 0
            return (source_priority(chunk), keyword_score, length_score)

        return sorted(chunks, key=score, reverse=True)[:limit]

    def field_claims(self, extraction: IndustryExtraction) -> dict[str, list[str]]:
        claims: dict[str, list[str]] = {
            "industry": [extraction.industry],
            "target_market": [extraction.target_market],
            "market_size": [extraction.market_size],
            "market_growth_rate": [extraction.market_growth_rate],
            "growth_drivers": extraction.growth_drivers,
            "customer_pain": extraction.customer_pain,
            "competition": extraction.competition,
            "regulation_or_policy": extraction.regulation_or_policy,
            "opportunity": extraction.opportunity,
            "risk": extraction.risk,
        }
        return {field: [claim for claim in values if claim] for field, values in claims.items()}

    @staticmethod
    def _guess_company(text: str) -> str:
        for pattern in COMPANY_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1).strip()[:50]
        return ""

    def _first_sentence_by_keywords(self, chunks: list[ParsedChunk], keywords: list[str], require_number: bool = False) -> str:
        matches = self._ranked_sentences(chunks, keywords, require_number=require_number)
        return matches[0] if matches else ""

    def _top_sentences(self, chunks: list[ParsedChunk], keywords: list[str], limit: int) -> list[str]:
        deduped = OrderedDict()
        for sentence in self._ranked_sentences(chunks, keywords):
            simplified = sentence.strip()
            if simplified and simplified not in deduped:
                deduped[simplified] = None
            if len(deduped) >= limit:
                break
        return list(deduped.keys())

    def _ranked_sentences(self, chunks: list[ParsedChunk], keywords: list[str], require_number: bool = False) -> list[str]:
        scored: list[tuple[int, str]] = []
        for chunk in chunks:
            for sentence in self._sentences(chunk.text):
                if self._looks_noisy(sentence):
                    continue
                lowered = sentence.lower()
                if require_number and not re.search(r"[\d,.]+ ?(%|조|억|만|billion|million|억원|조원)", lowered):
                    continue
                score = sum(1 for keyword in keywords if keyword.lower() in lowered)
                if score:
                    score += self._quality_bonus(sentence)
                    scored.append((score, sentence))
        scored.sort(key=lambda item: (item[0], -len(item[1])), reverse=True)
        return [sentence for _, sentence in scored]

    @staticmethod
    def _sentences(text: str) -> list[str]:
        rough = re.split(r"(?<=[.!?。])\s+|\n+|(?<=다\.)\s*|(?<=됨)\s+|(?<=음)\s+", text)
        sentences = []
        for item in rough:
            item = re.sub(r"\s+", " ", item).strip(" -•\t")
            if 18 <= len(item) <= 260:
                sentences.append(item)
        return sentences

    @staticmethod
    def _looks_noisy(sentence: str) -> bool:
        lowered = sentence.lower()
        if any(hint in lowered for hint in NOISE_HINTS):
            return True
        if sentence.count("|") >= 3 or sentence.count("•") >= 8:
            return True
        tokens = sentence.split()
        if len(tokens) >= 12:
            numeric_like = sum(1 for token in tokens if re.search(r"\d", token))
            if numeric_like / len(tokens) > 0.45:
                return True
        latin = sum(1 for char in sentence if char.isascii() and char.isalpha())
        korean = sum(1 for char in sentence if "가" <= char <= "힣")
        if latin > korean * 3 and len(sentence) > 80:
            return True
        return False

    @staticmethod
    def _quality_bonus(sentence: str) -> int:
        bonus = 0
        if any(marker in sentence for marker in ["시장", "산업", "고객", "경쟁", "리스크", "기회"]):
            bonus += 1
        if any(ending in sentence for ending in ["다.", "음", "됨", "필요", "예상", "판단"]):
            bonus += 1
        if 35 <= len(sentence) <= 180:
            bonus += 1
        return bonus
