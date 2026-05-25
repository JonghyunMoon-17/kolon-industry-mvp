from __future__ import annotations

from src.llm import BaseLLMProvider
from src.models import ParsedChunk


class DirectDraftGenerator:
    """Generate the report section directly from parsed source material.

    This path is intentionally closer to how a human would ask Claude: read the
    core materials, use samples only as style references, and write the section.
    """

    def generate(
        self,
        company_chunks: list[ParsedChunk],
        sample_chunks: list[ParsedChunk],
        provider: BaseLLMProvider,
    ) -> str:
        try:
            brief_prompt = self._build_brief_prompt(company_chunks)
            brief = provider.generate_text(brief_prompt)
            draft_prompt = self._build_draft_prompt(brief, company_chunks, sample_chunks)
            return provider.generate_text(draft_prompt)
        except Exception as exc:
            if "rate_limit" not in str(exc) and "rate limit" not in str(exc).lower():
                raise
            compact_prompt = self._build_compact_prompt(company_chunks, sample_chunks)
            return provider.generate_text(compact_prompt)

    def _build_brief_prompt(self, company_chunks: list[ParsedChunk]) -> str:
        return (
            "당신은 VC 투자심사보고서 작성을 위한 산업분석 리서치 애널리스트입니다.\n"
            "아래 회사 자료를 바탕으로 「산업 현황 및 분석」 섹션 작성에 필요한 분석 메모를 만드세요.\n\n"
            "작성 지침:\n"
            "- 회사 자료에 없는 사실을 만들지 마세요.\n"
            "- 제품 스펙 나열을 피하고 산업/시장/고객/경쟁/리스크 관점으로 재구성하세요.\n"
            "- 수치, 계약, 양산, 매출, 시장규모는 출처/확인 필요성을 구분하세요.\n"
            "- 출력은 보고서 초안 작성자가 바로 쓸 수 있게 구체적으로 쓰세요.\n\n"
            "반드시 다음 목차로 작성:\n"
            "1. 투자대상기업 및 산업 범위\n"
            "2. 시장 성장 배경\n"
            "3. 고객 니즈와 구매 전환 조건\n"
            "4. 경쟁환경 및 주요 플레이어\n"
            "5. 투자대상기업의 기회 요인\n"
            "6. 주요 리스크와 추가 확인사항\n"
            "7. 산업분석 초안에 반드시 반영할 문장 재료\n\n"
            "<company_materials>\n"
            f"{self._format_chunks(company_chunks, max_chars=18000, limit=24)}\n"
            "</company_materials>"
        )

    def _build_draft_prompt(self, brief: str, company_chunks: list[ParsedChunk], sample_chunks: list[ParsedChunk]) -> str:
        return (
            "당신은 한국 VC 투자심사보고서 작성자입니다.\n"
            "아래 분석 메모와 근거 일부, 그리고 샘플 보고서 문체를 참고하여 "
            "투심보고서의 「산업 현황 및 분석」 섹션 초안을 작성하세요.\n\n"
            "중요:\n"
            "- 샘플 보고서는 문체/구성 참고용입니다. 샘플의 회사명, 산업, 사실관계를 섞지 마세요.\n"
            "- 원문을 길게 복사하지 말고 투자심사보고서 문단으로 재작성하세요.\n"
            "- 기술 스펙은 필요한 경우에만 산업적 의미로 번역하세요.\n"
            "- 단정적 홍보 문구를 피하고 '~로 판단됨', '~로 예상됨', '~확인 필요' 같은 검토보고서 문체를 사용하세요.\n"
            "- 외부 검색 없이 입력자료 기준으로만 쓰세요.\n"
            "- 5~7개 문단으로 작성하세요.\n"
            "- 첫 문단은 산업 정의/범위, 마지막 문단은 기회와 리스크의 균형으로 마무리하세요.\n\n"
            "<analysis_brief>\n"
            f"{brief}\n"
            "</analysis_brief>\n\n"
            "<selected_evidence>\n"
            f"{self._format_chunks(company_chunks, max_chars=6000, limit=10)}\n"
            "</selected_evidence>\n\n"
            "<style_samples>\n"
            f"{self._format_chunks(sample_chunks, max_chars=3000, limit=4)}\n"
            "</style_samples>\n\n"
            "Markdown으로 '## 산업 현황 및 분석' 제목부터 출력하세요."
        )

    def _build_compact_prompt(self, company_chunks: list[ParsedChunk], sample_chunks: list[ParsedChunk]) -> str:
        return (
            "당신은 한국 VC 투자심사보고서 작성자입니다.\n"
            "아래 SolidVUE 회사 자료와 샘플 문체 일부를 참고하여 「산업 현황 및 분석」 초안을 바로 작성하세요.\n"
            "5~7개 문단, 투자검토 보고서 문체, 과장 금지, 수치/전망은 보수적으로 작성하세요.\n"
            "제품 스펙 나열이 아니라 산업/시장/고객/경쟁/기회/리스크 중심으로 쓰세요.\n\n"
            "<company_materials>\n"
            f"{self._format_chunks(company_chunks, max_chars=14000, limit=18)}\n"
            "</company_materials>\n\n"
            "<style_samples>\n"
            f"{self._format_chunks(sample_chunks, max_chars=2000, limit=3)}\n"
            "</style_samples>\n\n"
            "Markdown으로 '## 산업 현황 및 분석' 제목부터 출력하세요."
        )

    @staticmethod
    def _format_chunks(chunks: list[ParsedChunk], max_chars: int, limit: int = 52) -> str:
        selected = DirectDraftGenerator._select_chunks(chunks, limit=limit)
        blocks: list[str] = []
        used = 0
        for chunk in selected:
            text = chunk.text[:900]
            block = (
                f"[{chunk.chunk_id}] {chunk.source_file} "
                f"type={chunk.doc_type} page/slide={chunk.page_or_slide or '-'} "
                f"section={chunk.section_title or '-'}\n{text}"
            )
            if used + len(block) > max_chars:
                break
            blocks.append(block)
            used += len(block)
        return "\n\n---\n\n".join(blocks)

    @staticmethod
    def _select_chunks(chunks: list[ParsedChunk], limit: int = 52) -> list[ParsedChunk]:
        keywords = [
            "시장",
            "산업",
            "성장",
            "수요",
            "고객",
            "경쟁",
            "리스크",
            "기회",
            "양산",
            "계약",
            "매출",
            "OEM",
            "LiDAR",
            "라이다",
            "자율주행",
            "로봇",
        ]

        def priority(chunk: ParsedChunk) -> tuple[int, int, int]:
            name = chunk.source_file.lower()
            source_score = 0
            if "pitch" in name or "녹취" in name:
                source_score = 5
            elif "memo" in name or "딜메모" in name:
                source_score = 4
            elif chunk.doc_type == "docx":
                source_score = 3
            elif chunk.doc_type == "pdf":
                source_score = 2
            keyword_score = sum(chunk.text.count(keyword) for keyword in keywords)
            length_score = 1 if 120 <= len(chunk.text) <= 1500 else 0
            return (source_score, keyword_score, length_score)

        return sorted(chunks, key=priority, reverse=True)[:limit]
