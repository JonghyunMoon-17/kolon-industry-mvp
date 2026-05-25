# Kolon Investment Industry Analysis MVP

Streamlit prototype for generating the "산업 현황 및 분석" section of an investment committee report.

The MVP follows this flow:

```text
Document upload -> ParsedChunk normalization -> evidence retrieval -> structured extraction
-> mini KG -> rule validation -> industry analysis draft
```

The app is intentionally local-first for demos. It can use OpenAI/Claude later through the provider interface, but it also includes deterministic fallback extraction and lexical retrieval so the prototype runs without API keys.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

If optional parsers are not installed, TXT/MD files still work. PDF/DOCX/PPTX support improves when the dependencies in `requirements.txt` are installed.

## Claude API Mode

The default `heuristic-demo` mode is free and local, but draft quality is limited. For a stronger demo, use Claude Sonnet 4 (`claude-sonnet-4-20250514` by default):

1. Select `claude-api` in the Streamlit sidebar.
2. Paste your Anthropic API key in the password field.
3. Run the pipeline.

The key is used only for the current Streamlit session and is not written to project files. You can also set it as an environment variable:

```bash
export ANTHROPIC_API_KEY="your_key_here"
streamlit run app/streamlit_app.py
```

Claude is used for:

- `IndustryExtraction` structured JSON extraction
- final report-style draft generation

Rule-based evidence retrieval, mini KG generation, and validation still run locally.

For better writing quality, use `생성 방식 = direct-draft`. This sends selected company material chunks and sample style chunks directly to Claude and asks it to write the report section. Use `pipeline` when you want to demonstrate the structured extraction/KG/evidence flow.

Use `Claude model = auto` unless you have a known valid model ID. The app calls Anthropic's model-list endpoint and selects an available Sonnet model when possible. If Claude mode fails, the app now stops instead of showing a fallback draft as if it were a real Claude result.

## Project Structure

```text
app/streamlit_app.py          Streamlit demo UI
configs/                      Style guide, ontology, validation rules
src/models.py                 Pydantic schemas for all pipeline artifacts
src/parsing/                  Extension-based parsing into ParsedChunk
src/rag/                      ChromaDB adapter with lexical fallback
src/extraction/               Schema-bound extraction with heuristic fallback
src/ontology/                 Deterministic mini-KG builder
src/validation/               Rule-based evidence and missing-field checks
src/generation/               Evidence-aware draft generation
tests/                        Lightweight unit tests
```

## Notes

- The seven sample reports are style/structure guides, not supervised labels.
- The MVP does not include Neptune, OpenSearch, AWS deployment, RBAC, or a full GraphRAG stack.
- ChromaDB is optional. When unavailable, the app uses an in-memory lexical retriever with the same `EvidenceMap` output shape.
