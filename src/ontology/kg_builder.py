from __future__ import annotations

import re

from src.models import IndustryExtraction, KGEdge, KGNode, MiniKG


def slug(value: str) -> str:
    value = re.sub(r"\s+", "", value or "")
    value = re.sub(r"[^A-Za-z0-9가-힣_.:-]", "", value)
    return value[:64] or "unknown"


def build_mini_kg(extraction: IndustryExtraction) -> MiniKG:
    nodes: dict[str, KGNode] = {}
    edges: list[KGEdge] = []

    def add_node(prefix: str, type_: str, label: str) -> str:
        if not label:
            return ""
        node_id = f"{prefix}:{slug(label)}"
        nodes.setdefault(node_id, KGNode(id=node_id, type=type_, label=label))
        return node_id

    def add_edge(source: str, relation: str, target: str) -> None:
        if source and target:
            edges.append(KGEdge(source=source, relation=relation, target=target))

    company_id = add_node("company", "투자대상기업", extraction.company_name or "투자대상기업")
    industry_id = add_node("industry", "산업", extraction.industry)
    market_id = add_node("market", "시장", extraction.target_market)

    add_edge(company_id, "속한다", industry_id)
    add_edge(industry_id, "포함한다", market_id)

    if extraction.market_size:
        size_id = add_node("market_size", "시장규모", extraction.market_size)
        add_edge(market_id, "가진다", size_id)
    if extraction.market_growth_rate:
        rate_id = add_node("growth_rate", "성장률", extraction.market_growth_rate)
        add_edge(market_id, "가진다", rate_id)

    for value in extraction.growth_drivers:
        driver_id = add_node("driver", "성장요인", value)
        add_edge(market_id or industry_id, "가진다", driver_id)
    for value in extraction.customer_pain:
        pain_id = add_node("pain", "문제/니즈", value)
        add_edge(company_id, "해결한다", pain_id)
    for value in extraction.competition:
        comp_id = add_node("competition", "경쟁환경", value)
        add_edge(industry_id, "가진다", comp_id)
    for value in extraction.regulation_or_policy:
        policy_id = add_node("policy", "규제/정책", value)
        add_edge(market_id or industry_id, "영향을 받는다", policy_id)
    for value in extraction.opportunity:
        opp_id = add_node("opportunity", "사업기회", value)
        add_edge(industry_id or market_id, "만든다", opp_id)
    for value in extraction.risk:
        risk_id = add_node("risk", "리스크", value)
        add_edge(industry_id or market_id, "가진다", risk_id)

    return MiniKG(nodes=list(nodes.values()), edges=edges)
