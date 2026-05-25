from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PipelineDiagnostics:
    events: list[str] = field(default_factory=list)

    def add(self, message: str) -> None:
        self.events.append(message)
