from __future__ import annotations

from collections import Counter

from src.models import ParsedChunk


REPORT_PHRASES = ["판단됨", "예상됨", "해석됨", "검토", "필요", "가능성", "시사", "리스크"]


def analyze_style(chunks: list[ParsedChunk]) -> list[str]:
    """Extract lightweight style notes from sample investment-report sections.

    This is intentionally conservative: sample chunks should never feed company
    facts, only writing guidance.
    """

    if not chunks:
        return []

    text = "\n".join(chunk.text for chunk in chunks)
    notes = [
        f"샘플 {len({chunk.source_file for chunk in chunks})}개 파일, {len(chunks)}개 chunk를 문체 참고용으로만 사용함",
    ]

    phrase_counts = Counter({phrase: text.count(phrase) for phrase in REPORT_PHRASES})
    common = [phrase for phrase, count in phrase_counts.most_common() if count > 0]
    if common:
        notes.append("자주 보이는 보고서식 표현: " + ", ".join(common[:5]))

    if "리스크" in text or "위험" in text or "검토 필요" in text:
        notes.append("결론부에서는 기회 요인과 함께 리스크/검토 필요사항을 병기")
    if "%" in text or "억원" in text or "조원" in text:
        notes.append("수치 표현은 출처와 산정 기준을 함께 확인하는 방식으로 보수적으로 작성")
    notes.append("샘플 내용의 사실관계는 특정 회사 extraction에 사용하지 않음")
    return notes
