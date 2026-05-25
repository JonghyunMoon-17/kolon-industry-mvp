from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.extraction import IndustryExtractor
from src.extraction.style_analyzer import analyze_style
from src.generation import DirectDraftGenerator, ReportGenerator
from src.llm import ClaudeProvider
from src.ontology import build_mini_kg
from src.parsing import parse_uploaded_file
from src.pipeline_state import PipelineDiagnostics
from src.rag import EvidenceRetriever
from src.utils import dump_json, load_yaml
from src.validation import IndustryValidator


st.set_page_config(page_title="Kolon Industry Analysis MVP", layout="wide")


def make_provider(provider_name: str, api_key: str, model: str):
    if provider_name == "claude-api":
        secret_key = ""
        try:
            secret_key = st.secrets.get("ANTHROPIC_API_KEY", "")
        except Exception:
            secret_key = ""
        return ClaudeProvider(api_key=api_key.strip() or secret_key or None, model=model)
    return None


def run_pipeline(company_files, sample_files, provider_name: str, api_key: str, model: str, top_k: int, generation_mode: str):
    diagnostics = PipelineDiagnostics()
    chunks = []
    sample_chunks = []
    parse_errors = []
    for uploaded_file in company_files:
        try:
            chunks.extend(parse_uploaded_file(uploaded_file))
        except Exception as exc:
            parse_errors.append(f"{uploaded_file.name}: {exc}")
    for uploaded_file in sample_files:
        try:
            sample_chunks.extend(parse_uploaded_file(uploaded_file))
        except Exception as exc:
            parse_errors.append(f"{uploaded_file.name}: {exc}")

    llm_provider = None
    provider_error = ""
    try:
        llm_provider = make_provider(provider_name, api_key, model)
        if llm_provider is not None:
            diagnostics.add(f"Claude provider initialized: {model}")
    except Exception as exc:
        provider_error = str(exc)
        diagnostics.add(f"Claude provider initialization failed: {exc}")

    extractor = IndustryExtractor()
    extraction_provider = None if generation_mode == "direct-draft" else llm_provider
    extraction = extractor.extract(chunks, provider=extraction_provider, diagnostics=diagnostics) if chunks else extractor.extract([])
    retriever = EvidenceRetriever(chunks, persist_dir=str(ROOT / "data/vector_db/chroma"))
    evidence_map = retriever.build_evidence_map(extractor.field_claims(extraction), top_k=top_k)
    mini_kg = build_mini_kg(extraction)
    validation = IndustryValidator().validate(extraction, evidence_map)
    style_notes = analyze_style(sample_chunks)
    if generation_mode == "direct-draft" and llm_provider is not None:
        try:
            draft = DirectDraftGenerator().generate(chunks, sample_chunks, llm_provider)
            diagnostics.add("Claude direct draft generation succeeded")
        except Exception as exc:
            diagnostics.add(f"Claude direct draft generation failed; template fallback used: {exc}")
            draft = ReportGenerator().generate(extraction, evidence_map, validation, style_notes=style_notes)
    else:
        draft = ReportGenerator().generate(
            extraction,
            evidence_map,
            validation,
            style_notes=style_notes,
            provider=llm_provider,
            diagnostics=diagnostics,
        )
    return {
        "chunks": chunks,
        "sample_chunks": sample_chunks,
        "style_notes": style_notes,
        "parse_errors": parse_errors,
        "extraction": extraction,
        "evidence_map": evidence_map,
        "mini_kg": mini_kg,
        "validation": validation,
        "draft": draft,
        "provider": provider_name,
        "model": model,
        "provider_error": provider_error,
        "used_llm": llm_provider is not None,
        "generation_mode": generation_mode,
        "diagnostics": diagnostics.events,
    }


def render_json_download(label: str, filename: str, data) -> None:
    payload = dump_json(data)
    st.download_button(label, payload, file_name=filename, mime="application/json")


st.title("코오롱인베스트먼트 산업분석 MVP")

with st.sidebar:
    st.header("입력")
    company_files = st.file_uploader(
        "회사 자료: IR Deck, 딜메모, IR 녹취",
        type=["pdf", "pptx", "docx", "txt", "md"],
        accept_multiple_files=True,
        key="company_files",
    )
    sample_files = st.file_uploader(
        "샘플 7개: 산업분석 문체/구조 참고",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        key="sample_files",
    )
    provider = st.selectbox("LLM provider", ["heuristic-demo", "claude-api"])
    generation_mode = st.selectbox(
        "생성 방식",
        ["direct-draft", "pipeline"],
        help="direct-draft는 Claude에게 회사자료를 바로 읽혀 초안을 쓰게 합니다. pipeline은 구조화 추출/KG/evidence 흐름을 강조합니다.",
    )
    api_key = ""
    claude_model = "auto"
    if provider == "claude-api":
        claude_model = st.selectbox(
            "Claude model",
            [
                "auto",
                "claude-sonnet-4-6",
                "claude-sonnet-4-5",
                "claude-sonnet-4-20250514",
                "claude-3-5-haiku-20241022",
                "claude-3-haiku-20240307",
            ],
            help="auto는 API 키로 사용 가능한 모델을 조회한 뒤 Sonnet 계열을 우선 선택합니다.",
        )
        api_key = st.text_input("Anthropic API key", type="password", help="세션에서만 사용되며 코드/파일에 저장하지 않습니다.")
    top_k = st.slider("근거 chunk 수", min_value=1, max_value=5, value=3)
    run = st.button("실행", type="primary", use_container_width=True)

    st.caption("현재 데모는 API 키 없이도 동작하는 heuristic fallback을 사용합니다.")

if run:
    st.session_state["result"] = run_pipeline(company_files, sample_files, provider, api_key, claude_model, top_k, generation_mode)

result = st.session_state.get("result")

if not result:
    st.info("좌측에서 파일을 업로드한 뒤 실행하세요. TXT/MD 파일만으로도 전체 흐름을 확인할 수 있습니다.")
    st.stop()

if result["parse_errors"]:
    st.warning("일부 파일 파싱에 실패했습니다.")
    for error in result["parse_errors"]:
        st.write(f"- {error}")

if result.get("provider_error"):
    st.warning(f"LLM provider를 사용할 수 없어 heuristic fallback으로 실행했습니다: {result['provider_error']}")
elif result.get("used_llm"):
    if any("failed" in event for event in result.get("diagnostics", [])):
        st.warning("Claude API를 시도했지만 일부 단계가 fallback으로 처리되었습니다.")
    else:
        st.success("Claude API를 사용해 구조화 추출과 최종 초안을 생성했습니다.")

failed_events = [event for event in result.get("diagnostics", []) if "failed" in event]
if failed_events:
    for event in failed_events:
        st.error(event)
    if result.get("provider") == "claude-api":
        st.stop()

if result.get("diagnostics"):
    with st.expander("실행 진단 로그", expanded=bool(failed_events)):
        for event in result["diagnostics"]:
            st.write(f"- {event}")

chunks = result["chunks"]
sample_chunks = result["sample_chunks"]
style_notes = result["style_notes"]
extraction = result["extraction"]
evidence_map = result["evidence_map"]
mini_kg = result["mini_kg"]
validation = result["validation"]
draft = result["draft"]

tab_extract, tab_kg, tab_evidence, tab_validation, tab_draft = st.tabs(
    ["추출 결과", "온톨로지/KG", "근거 검색", "검증 결과", "최종 초안"]
)

with tab_extract:
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.subheader("IndustryExtraction")
        st.json(extraction.model_dump())
        render_json_download("extraction_result.json 다운로드", "extraction_result.json", extraction)
    with col_b:
        st.subheader("ParsedChunk")
        st.metric("company chunks", len(chunks))
        st.metric("sample chunks", len(sample_chunks))
        if chunks:
            st.json(chunks[0].model_dump())
        if style_notes:
            st.subheader("Style notes")
            for note in style_notes:
                st.write(f"- {note}")

with tab_kg:
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("industry_ontology.yaml")
        st.json(load_yaml(ROOT / "configs/industry_ontology.yaml"))
    with col_b:
        st.subheader("mini_kg.json")
        st.json(mini_kg.model_dump())
        render_json_download("mini_kg.json 다운로드", "mini_kg.json", mini_kg)

    try:
        import networkx as nx
        from pyvis.network import Network

        graph = nx.DiGraph()
        for node in mini_kg.nodes:
            graph.add_node(node.id, label=node.label, title=node.type)
        for edge in mini_kg.edges:
            graph.add_edge(edge.source, edge.target, label=edge.relation)
        net = Network(height="460px", width="100%", directed=True, bgcolor="#ffffff")
        net.from_nx(graph)
        html = net.generate_html(notebook=False)
        st.components.v1.html(html, height=480)
    except Exception as exc:
        st.caption(f"그래프 시각화 라이브러리를 사용할 수 없습니다: {exc}")

with tab_evidence:
    st.subheader("evidence_map.json")
    st.json(evidence_map.model_dump())
    render_json_download("evidence_map.json 다운로드", "evidence_map.json", evidence_map)
    for item in evidence_map.items:
        with st.expander(f"{item.field}: {item.claim[:80]}"):
            if not item.evidence_chunks:
                st.write("연결된 근거 chunk가 없습니다.")
            for chunk in item.evidence_chunks:
                st.markdown(f"**{chunk.evidence_id}** · {chunk.source_file} · {chunk.page_or_slide or '-'} · score {chunk.score}")
                st.write(chunk.text)

with tab_validation:
    st.subheader("validation_report.json")
    st.json(validation.model_dump())
    render_json_download("validation_report.json 다운로드", "validation_report.json", validation)

    if validation.missing_fields:
        st.warning("누락 필드: " + ", ".join(validation.missing_fields))
    if validation.warnings:
        for warning in validation.warnings:
            if warning.severity == "high":
                st.error(f"[{warning.rule}] {warning.message}")
            else:
                st.warning(f"[{warning.rule}] {warning.message}")
    else:
        st.success("검증 경고가 없습니다.")

with tab_draft:
    st.subheader("산업 현황 및 분석 초안")
    st.markdown(draft)
    st.download_button("Markdown 다운로드", draft, file_name="industry_analysis_draft.md", mime="text/markdown")
    st.download_button("Word-compatible 텍스트 다운로드", draft, file_name="industry_analysis_draft.doc", mime="application/msword")
