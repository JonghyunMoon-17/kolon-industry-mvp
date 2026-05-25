from __future__ import annotations

import re

from src.models import EvidenceMap, IndustryExtraction, ValidationReport, ValidationWarning


class IndustryValidator:
    REQUIRED_FIELDS = [
        "industry",
        "target_market",
        "growth_drivers",
        "customer_pain",
        "competition",
        "opportunity",
        "risk",
    ]

    ASSERTIVE_PATTERNS = ["반드시", "확실히", "무조건", "압도적", "유일", "독점"]

    def validate(self, extraction: IndustryExtraction, evidence_map: EvidenceMap) -> ValidationReport:
        passed: list[str] = []
        warnings: list[ValidationWarning] = []
        missing_fields: list[str] = []

        for field in self.REQUIRED_FIELDS:
            value = getattr(extraction, field)
            if not value:
                missing_fields.append(field)

        if extraction.market_size or extraction.market_growth_rate:
            has_market_evidence = evidence_map.has_evidence("market_size") or evidence_map.has_evidence("market_growth_rate")
            if has_market_evidence:
                passed.append("market_size_evidence")
            else:
                warnings.append(
                    ValidationWarning(
                        rule="market_size_evidence",
                        severity="high",
                        field="market_size",
                        message="시장규모 또는 성장률 수치가 있으나 연결된 근거 chunk가 없습니다.",
                    )
                )

        if len(extraction.growth_drivers) >= 2:
            passed.append("growth_driver_required")
        else:
            warnings.append(
                ValidationWarning(
                    rule="growth_driver_required",
                    severity="medium",
                    field="growth_drivers",
                    message="성장요인은 최소 2개 이상 제시하는 것이 좋습니다.",
                )
            )

        if extraction.competition:
            passed.append("competition_required")
        else:
            warnings.append(
                ValidationWarning(
                    rule="competition_required",
                    severity="medium",
                    field="competition",
                    message="경쟁환경 또는 주요 플레이어 정보가 부족합니다.",
                )
            )

        if extraction.opportunity and evidence_map.has_evidence("opportunity"):
            passed.append("opportunity_link_required")
        else:
            warnings.append(
                ValidationWarning(
                    rule="opportunity_link_required",
                    severity="high",
                    field="opportunity",
                    message="산업 변화가 투자대상기업의 기회와 연결되는 근거가 부족합니다.",
                )
            )

        if extraction.risk:
            passed.append("risk_required")
        else:
            warnings.append(
                ValidationWarning(
                    rule="risk_required",
                    severity="medium",
                    field="risk",
                    message="산업/시장 관점의 리스크 또는 검토 필요사항이 부족합니다.",
                )
            )

        unsupported = self._unsupported_assertive_claims(extraction, evidence_map)
        if unsupported:
            warnings.append(
                ValidationWarning(
                    rule="no_unsupported_claim",
                    severity="high",
                    message=f"근거가 약한 확정적 표현 가능성이 있습니다: {', '.join(unsupported[:3])}",
                )
            )
        else:
            passed.append("no_unsupported_claim")

        return ValidationReport(passed=passed, warnings=warnings, missing_fields=missing_fields)

    def _unsupported_assertive_claims(self, extraction: IndustryExtraction, evidence_map: EvidenceMap) -> list[str]:
        claims = []
        for field, value in extraction.model_dump().items():
            values = value if isinstance(value, list) else [value]
            for item in values:
                if not isinstance(item, str):
                    continue
                if any(pattern in item for pattern in self.ASSERTIVE_PATTERNS) and not evidence_map.has_evidence(field):
                    claims.append(item)
                if re.search(r"\d+ ?%", item) and not evidence_map.has_evidence(field):
                    claims.append(item)
        return claims
