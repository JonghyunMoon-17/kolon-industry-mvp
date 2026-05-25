from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anthropic import Anthropic

from src.models import ElementType
from src.models import ParsedChunk
from src.parsing import parse_path
from src.utils import split_text


DEFAULT_MODEL = "claude-sonnet-4-6"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a VC industry-analysis draft directly with Claude.")
    parser.add_argument("--company", action="append", required=True, help="Company source file path. Repeatable.")
    parser.add_argument("--sample", action="append", default=[], help="Style sample file path. Repeatable.")
    parser.add_argument("--out", default="outputs/solidvue_industry_analysis.md")
    parser.add_argument("--brief-out", default="outputs/solidvue_industry_brief.md")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY is required")

    client = Anthropic(api_key=api_key)
    company_chunks = parse_files(args.company)
    sample_chunks = parse_files(args.sample)

    print(f"Parsed company chunks: {len(company_chunks)}")
    print(f"Parsed sample chunks: {len(sample_chunks)}")

    brief_prompt = build_brief_prompt(company_chunks)
    brief = call_claude(client, args.model, brief_prompt, max_tokens=5000, temperature=0.1)
    Path(args.brief_out).write_text(brief, encoding="utf-8")

    draft_prompt = build_draft_prompt(brief, company_chunks, sample_chunks)
    draft = call_claude(client, args.model, draft_prompt, max_tokens=6000, temperature=0.2)
    Path(args.out).write_text(draft, encoding="utf-8")
    print(f"Wrote brief: {args.brief_out}")
    print(f"Wrote draft: {args.out}")


def parse_files(paths: list[str]) -> list[ParsedChunk]:
    chunks: list[ParsedChunk] = []
    for path in paths:
        p = Path(path)
        if p.suffix.lower() == ".pdf":
            parsed = parse_pdf_fast(p)
        else:
            parsed = parse_path(p)
        chunks.extend(parsed)
        print(f"Parsed {len(parsed):>3} chunks from {p.name}", flush=True)
    return chunks


def parse_pdf_fast(path: Path) -> list[ParsedChunk]:
    import fitz

    chunks: list[ParsedChunk] = []
    with fitz.open(path) as doc:
        for page_index, page in enumerate(doc, start=1):
            text = page.get_text("text")
            for local_index, part in enumerate(split_text(text, max_chars=1200, overlap=80)):
                chunks.append(
                    ParsedChunk(
                        chunk_id=f"pdf_{path.stem}_{page_index}_{local_index}",
                        source_file=path.name,
                        doc_type="pdf",
                        page_or_slide=str(page_index),
                        section_title=f"Page {page_index}",
                        text=part,
                        element_type=ElementType.TEXT,
                        metadata={"parser": "fitz_fast_text", "page": page_index},
                    )
                )
    return chunks


def call_claude(client: Anthropic, model: str, prompt: str, max_tokens: int, temperature: float) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
    return "\n".join(parts).strip()


def build_brief_prompt(company_chunks: list[ParsedChunk]) -> str:
    materials = format_chunks(select_company_chunks(company_chunks), max_chars=42000)
    return f"""
당신은 VC 투자심사보고서 작성을 위한 산업분석 리서치 애널리스트입니다.
아래 회사 자료는 SolidVue의 IR Deck, 딜메모, 피칭 메모에서 추출한 텍스트입니다.

목표:
투심보고서의 「산업 현황 및 분석」 섹션을 쓰기 전에 필요한 핵심 사실과 판단 근거를 정리하세요.

작성 지침:
- 회사 자료에 없는 사실을 만들지 마세요.
- 제품 스펙 나열을 피하고, 산업/시장/고객/경쟁/리스크 관점으로 재구성하세요.
- LiDAR, 로봇/자율주행, OEM, 중국 경쟁사, SOSLAB/Sony/Adaps/Waymo 등 자료상 핵심 키워드는 맥락을 살려 정리하세요.
- 수치, 계약, 양산, 매출, 시장규모는 출처/확인 필요성을 구분하세요.
- 출력은 Markdown bullet 중심으로 간결하되, 보고서 초안 작성자가 바로 쓸 수 있게 충분히 구체적으로 쓰세요.

반드시 다음 목차로 작성:
1. 투자대상기업 및 산업 범위
2. 시장 성장 배경
3. 고객 니즈와 구매 전환 조건
4. 경쟁환경 및 주요 플레이어
5. 투자대상기업의 기회 요인
6. 주요 리스크와 추가 확인사항
7. 산업분석 초안에 반드시 반영할 문장 재료

<company_materials>
{materials}
</company_materials>
""".strip()


def build_draft_prompt(brief: str, company_chunks: list[ParsedChunk], sample_chunks: list[ParsedChunk]) -> str:
    sample_text = format_chunks(select_sample_chunks(sample_chunks), max_chars=10000)
    evidence_text = format_chunks(select_evidence_chunks(company_chunks), max_chars=10000)
    return f"""
당신은 한국 VC 투자심사보고서 작성자입니다.
아래 분석 메모와 근거 일부, 그리고 샘플 보고서 문체를 참고하여 SolidVue 투자검토보고서의 「산업 현황 및 분석」 섹션 초안을 작성하세요.

중요:
- 샘플 보고서는 문체/구성 참고용입니다. 샘플의 회사명, 산업, 사실관계를 SolidVue에 섞지 마세요.
- 원문을 길게 복사하지 말고, 투자심사보고서 문단으로 재작성하세요.
- IR의 기술 스펙은 필요한 경우에만 산업적 의미로 번역하세요.
- 단정적 홍보 문구를 피하고 "~로 판단됨", "~로 예상됨", "~확인 필요" 같은 검토보고서 문체를 사용하세요.
- 외부 검색 없이 입력자료 기준으로만 쓰세요.
- 5~7개 문단으로 작성하세요.
- 첫 문단은 산업 정의/범위, 마지막 문단은 기회와 리스크의 균형으로 마무리하세요.
- 제목은 "## 산업 현황 및 분석"만 사용하세요.

<analysis_brief>
{brief}
</analysis_brief>

<selected_evidence>
{evidence_text}
</selected_evidence>

<style_samples>
{sample_text}
</style_samples>
""".strip()


def select_company_chunks(chunks: list[ParsedChunk], limit: int = 52) -> list[ParsedChunk]:
    keywords = [
        "시장",
        "산업",
        "LiDAR",
        "라이다",
        "OEM",
        "고객",
        "경쟁",
        "SOSLAB",
        "Sony",
        "Adaps",
        "Waymo",
        "중국",
        "양산",
        "계약",
        "매출",
        "성장",
        "리스크",
        "기회",
        "자율주행",
        "로봇",
    ]

    def score(chunk: ParsedChunk) -> tuple[int, int, int]:
        name = chunk.source_file.lower()
        source = 0
        if "pitch" in name:
            source = 5
        elif "딜메모" in name or "memo" in name:
            source = 5
        elif chunk.doc_type == "docx":
            source = 4
        elif chunk.doc_type == "pdf":
            source = 2
        keyword = sum(chunk.text.count(k) for k in keywords)
        readable = 1 if 80 <= len(chunk.text) <= 1800 and chunk.text.count("|") < 8 else 0
        return (source, keyword, readable)

    ranked = sorted(chunks, key=score, reverse=True)
    return ranked[:limit]


def select_evidence_chunks(chunks: list[ParsedChunk], limit: int = 20) -> list[ParsedChunk]:
    return select_company_chunks(chunks, limit=limit)


def select_sample_chunks(chunks: list[ParsedChunk], limit: int = 16) -> list[ParsedChunk]:
    def score(chunk: ParsedChunk) -> tuple[int, int]:
        keyword = sum(chunk.text.count(k) for k in ["시장", "산업", "성장", "리스크", "판단", "예상", "검토"])
        readable = 1 if 120 <= len(chunk.text) <= 1800 else 0
        return (keyword, readable)

    return sorted(chunks, key=score, reverse=True)[:limit]


def format_chunks(chunks: list[ParsedChunk], max_chars: int) -> str:
    blocks: list[str] = []
    used = 0
    for chunk in chunks:
        text = clean_text(chunk.text)
        if not text:
            continue
        block = (
            f"[{chunk.chunk_id}] {chunk.source_file} "
            f"type={chunk.doc_type} page/slide={chunk.page_or_slide or '-'} "
            f"section={chunk.section_title or '-'}\n{text[:1800]}"
        )
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n---\n\n".join(blocks)


def clean_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = " ".join(line.split())
        if not line:
            continue
        if line.count("|") >= 8:
            continue
        lines.append(line)
    return "\n".join(lines)


if __name__ == "__main__":
    main()
