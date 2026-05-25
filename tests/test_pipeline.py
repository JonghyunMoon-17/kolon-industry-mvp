from pathlib import Path

from src.extraction import IndustryExtractor
from src.generation import ReportGenerator
from src.ontology import build_mini_kg
from src.parsing import parse_path
from src.rag import EvidenceRetriever
from src.validation import IndustryValidator


def test_txt_pipeline_end_to_end(tmp_path: Path):
    sample = tmp_path / "pitch_transcript.txt"
    sample.write_text(
        """
        A기업은 AI 반도체 산업에서 데이터센터 AI 가속기 시장을 목표로 한다.
        데이터센터 AI 가속기 시장은 AI 서버 수요 증가와 클라우드 인프라 확대에 따라 성장하고 있다.
        시장규모는 2028년 10조원 수준으로 제시되며 CAGR 25% 성장이 예상된다.
        고객은 기존 GPU 비용 부담과 전력 소모 문제를 겪고 있다.
        경쟁환경은 NVIDIA와 ASIC 스타트업 중심으로 형성되어 있다.
        저전력 특화 칩 수요 증가는 사업기회가 될 수 있다.
        다만 양산 검증 필요와 대형 경쟁사 진입은 리스크로 판단된다.
        """,
        encoding="utf-8",
    )

    chunks = parse_path(sample)
    assert chunks
    assert chunks[0].source_file == "pitch_transcript.txt"
    assert chunks[0].metadata["parser"] == "plain_text"

    extractor = IndustryExtractor()
    extraction = extractor.extract(chunks)
    assert extraction.growth_drivers
    assert extraction.risk

    retriever = EvidenceRetriever(chunks)
    evidence_map = retriever.build_evidence_map(extractor.field_claims(extraction))
    assert evidence_map.items

    kg = build_mini_kg(extraction)
    assert kg.nodes
    assert kg.edges

    report = IndustryValidator().validate(extraction, evidence_map)
    assert "growth_driver_required" in report.passed

    draft = ReportGenerator().generate(extraction, evidence_map, report)
    assert "산업 현황 및 분석" in draft
    assert "근거:" in draft


def test_validation_warns_on_missing_evidence(tmp_path: Path):
    sample = tmp_path / "memo.txt"
    sample.write_text("시장규모는 100조원이며 성장률은 50%이다.", encoding="utf-8")
    chunks = parse_path(sample)
    extractor = IndustryExtractor()
    extraction = extractor.extract(chunks)
    empty_evidence = EvidenceRetriever([]).build_evidence_map({})
    report = IndustryValidator().validate(extraction, empty_evidence)
    assert any(w.rule == "market_size_evidence" for w in report.warnings)
