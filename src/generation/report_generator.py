from __future__ import annotations

import re

from src.llm import BaseLLMProvider
from src.models import EvidenceMap, IndustryExtraction, ValidationReport
from src.pipeline_state import PipelineDiagnostics


class ReportGenerator:
    def generate(
        self,
        extraction: IndustryExtraction,
        evidence_map: EvidenceMap,
        validation: ValidationReport,
        style_notes: list[str] | None = None,
        provider: BaseLLMProvider | None = None,
        diagnostics: PipelineDiagnostics | None = None,
    ) -> str:
        if provider is not None:
            try:
                draft = self._llm_generate(extraction, evidence_map, validation, style_notes or [], provider)
                if diagnostics:
                    diagnostics.add("Claude draft generation succeeded")
                return draft
            except Exception as exc:
                if diagnostics:
                    diagnostics.add(f"Claude draft generation failed; template fallback used: {exc}")

        evidence_lookup = self._evidence_lookup(evidence_map)

        paragraphs = [
            self._industry_paragraph(extraction, evidence_lookup),
            self._growth_paragraph(extraction, evidence_lookup),
            self._customer_paragraph(extraction, evidence_lookup),
            self._competition_paragraph(extraction, evidence_lookup),
            self._opportunity_risk_paragraph(extraction, evidence_lookup, validation),
        ]
        body = "\n\n".join(p for p in paragraphs if p)
        style_text = ""
        if style_notes:
            lines = "\n".join(f"- {note}" for note in style_notes[:6])
            style_text = f"\n\n### 스타일 참고 메모\n{lines}"

        warning_text = ""
        if validation.warnings:
            warning_lines = "\n".join(f"- [{w.severity}] {w.message}" for w in validation.warnings)
            warning_text = f"\n\n### 검증 메모\n{warning_lines}"
        return f"## 산업 현황 및 분석\n\n{body}{style_text}{warning_text}\n"

    def _llm_generate(
        self,
        extraction: IndustryExtraction,
        evidence_map: EvidenceMap,
        validation: ValidationReport,
        style_notes: list[str],
        provider: BaseLLMProvider,
    ) -> str:
        evidence_brief = []
        for item in evidence_map.items[:18]:
            chunks = item.evidence_chunks[:2]
            refs = "; ".join(
                f"{chunk.evidence_id} {chunk.source_file} p/slide {chunk.page_or_slide or '-'}: {chunk.text[:320]}"
                for chunk in chunks
            )
            evidence_brief.append(f"- {item.field}: {item.claim}\n  evidence: {refs}")

        warnings = [f"[{warning.severity}] {warning.message}" for warning in validation.warnings]
        prompt = (
            "당신은 VC 투자심사보고서 작성 보조 에이전트입니다.\n"
            "아래 구조화 추출 결과와 근거를 바탕으로 '산업 현황 및 분석' 섹션 초안을 작성하세요.\n\n"
            "작성 규칙:\n"
            "- 한국어 투자심사보고서 문체로 작성하세요.\n"
            "- 원문을 길게 복사하지 말고, 산업/시장 관점으로 압축하세요.\n"
            "- 제품 홍보문처럼 쓰지 마세요.\n"
            "- 근거 없는 수치와 전망은 단정하지 마세요.\n"
            "- 문단 끝에는 참고한 evidence id를 괄호로 표시하세요.\n"
            "- 구성은 산업 정의/시장 배경 → 성장요인 → 고객 니즈 → 경쟁환경 → 기회/리스크 순서로 하세요.\n\n"
            f"<extraction>\n{extraction.model_dump_json(indent=2)}\n</extraction>\n\n"
            f"<evidence>\n{chr(10).join(evidence_brief)}\n</evidence>\n\n"
            f"<validation_warnings>\n{chr(10).join(warnings) if warnings else '없음'}\n</validation_warnings>\n\n"
            f"<style_notes>\n{chr(10).join(style_notes) if style_notes else '기본 투자검토 보고서식 문체 사용'}\n</style_notes>\n\n"
            "Markdown으로 제목 '## 산업 현황 및 분석'부터 출력하세요."
        )
        return provider.generate_text(prompt)

    def _industry_paragraph(self, extraction: IndustryExtraction, evidence: dict[str, str]) -> str:
        industry = self._clean_claim(extraction.industry, fallback="해당 산업")
        market = self._clean_claim(extraction.target_market, fallback="대상 시장")
        market_data = []
        if extraction.market_size:
            market_data.append(f"시장규모는 {self._clean_claim(extraction.market_size)}")
        if extraction.market_growth_rate:
            market_data.append(f"성장률은 {self._clean_claim(extraction.market_growth_rate)}")
        metric_sentence = ""
        if market_data:
            metric_sentence = f" 문서상 확인되는 {' 및 '.join(market_data)}로 제시되어 있으나, 수치 해석은 출처와 산정 기준 확인이 필요함."
        return f"투자대상기업은 {industry} 영역에서 {market}을 중심으로 사업 기회를 검토할 수 있음.{metric_sentence} {evidence.get('industry') or evidence.get('target_market') or ''}".strip()

    def _growth_paragraph(self, extraction: IndustryExtraction, evidence: dict[str, str]) -> str:
        if not extraction.growth_drivers:
            return "시장 성장요인은 입력 문서에서 충분히 구조화되지 않았으며, 추가 시장 자료 확인이 필요함."
        drivers = ", ".join(self._clean_list(extraction.growth_drivers, limit=3))
        return f"시장 성장 배경으로는 {drivers} 등이 확인됨. 이는 산업 수요가 단기적 이벤트보다 구조적 변화와 연결될 가능성을 시사하나, 외부 시장자료를 통한 보강 검증이 필요함. {evidence.get('growth_drivers', '')}".strip()

    def _customer_paragraph(self, extraction: IndustryExtraction, evidence: dict[str, str]) -> str:
        if not extraction.customer_pain:
            return "고객 문제/니즈는 현재 입력 자료만으로 명확히 특정되지 않아, 고객 인터뷰 또는 피칭 Q&A 근거 보강이 필요함."
        pains = ", ".join(self._clean_list(extraction.customer_pain, limit=3))
        return f"고객 측면에서는 {pains} 등이 주요 문제로 제시됨. 해당 니즈가 반복적으로 확인될 경우, 시장 수요의 실재성과 구매 전환 가능성을 판단하는 핵심 근거가 될 수 있음. {evidence.get('customer_pain', '')}".strip()

    def _competition_paragraph(self, extraction: IndustryExtraction, evidence: dict[str, str]) -> str:
        if not extraction.competition:
            return "경쟁환경은 아직 충분히 식별되지 않았으며, 주요 플레이어와 대체재 비교가 후속 검토 항목임."
        competition = ", ".join(self._clean_list(extraction.competition, limit=3))
        return f"경쟁환경 측면에서는 {competition} 등이 확인됨. 따라서 투자대상기업의 포지셔닝은 단순 시장 진입 여부보다 차별화 요소와 기존 플레이어 대비 우위의 지속 가능성을 중심으로 검토할 필요가 있음. {evidence.get('competition', '')}".strip()

    def _opportunity_risk_paragraph(self, extraction: IndustryExtraction, evidence: dict[str, str], validation: ValidationReport) -> str:
        opportunity = ", ".join(self._clean_list(extraction.opportunity, limit=3)) if extraction.opportunity else "산업 변화와 연결되는 사업기회"
        risk = ", ".join(self._clean_list(extraction.risk, limit=3)) if extraction.risk else "시장/실행 리스크"
        caution = " 검증 경고가 있는 항목은 보수적으로 해석해야 함." if validation.warnings else ""
        return f"종합하면 {opportunity}는 투자대상기업에 기회 요인으로 작용할 수 있으나, {risk}는 주요 검토 필요사항으로 남아 있음.{caution} {evidence.get('opportunity') or evidence.get('risk') or ''}".strip()

    @staticmethod
    def _evidence_lookup(evidence_map: EvidenceMap) -> dict[str, str]:
        lookup: dict[str, str] = {}
        for item in evidence_map.items:
            if item.evidence_chunks and item.field not in lookup:
                ids = ", ".join(chunk.evidence_id for chunk in item.evidence_chunks[:2])
                lookup[item.field] = f"(근거: {ids})"
        return lookup

    @classmethod
    def _clean_list(cls, values: list[str], limit: int) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            item = cls._clean_claim(value)
            if item and item not in cleaned:
                cleaned.append(item)
            if len(cleaned) >= limit:
                break
        return cleaned or ["입력 자료상 확인되는 관련 요인"]

    @staticmethod
    def _clean_claim(value: str, fallback: str = "") -> str:
        if not value:
            return fallback
        text = re.sub(r"\s+", " ", value).strip()
        text = re.sub(r"\(근거:.*?\)", "", text).strip()
        text = re.sub(r"^[\"'•\-\d\.\)\s]+", "", text)
        parts = re.split(r"\s*[|/•]\s*|(?<=다\.)\s+|(?<=됨)\s+", text)
        candidates = [part.strip(" ,;:") for part in parts if 8 <= len(part.strip()) <= 120]
        if candidates:
            candidates.sort(key=lambda item: (ReportGenerator._claim_quality(item), -len(item)), reverse=True)
            text = candidates[0]
        if len(text) > 120:
            text = text[:117].rstrip() + "..."
        return text or fallback

    @staticmethod
    def _claim_quality(text: str) -> int:
        score = 0
        for keyword in ["시장", "고객", "경쟁", "기회", "리스크", "성장", "수요", "사업화", "양산"]:
            if keyword in text:
                score += 1
        if re.search(r"[A-Za-z]{8,}", text):
            score -= 1
        if sum(1 for char in text if char.isdigit()) > len(text) * 0.25:
            score -= 2
        return score
