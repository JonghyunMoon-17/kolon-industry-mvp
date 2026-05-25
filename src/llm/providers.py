from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any


class BaseLLMProvider(ABC):
    name: str

    @abstractmethod
    def generate_structured(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def generate_text(self, prompt: str) -> str:
        raise NotImplementedError


class HeuristicProvider(BaseLLMProvider):
    name = "heuristic-demo"

    def generate_structured(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        return {}

    def generate_text(self, prompt: str) -> str:
        return prompt


class ClaudeProvider(BaseLLMProvider):
    name = "claude"

    def __init__(self, api_key: str | None = None, model: str = "auto"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for Claude provider")
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError("Claude provider requires the anthropic package. Install requirements.txt.") from exc
        self.client = Anthropic(api_key=self.api_key)
        self.model = self._resolve_model(model)

    def _resolve_model(self, requested_model: str) -> str:
        if requested_model and requested_model != "auto":
            return requested_model
        try:
            models = list(self.client.models.list().data)
            ids = [model.id for model in models]
        except Exception as exc:
            raise RuntimeError(f"Could not list available Anthropic models: {exc}") from exc
        preferred = [
            "claude-sonnet-4-6",
            "claude-sonnet-4-5",
            "claude-sonnet-4-20250514",
            "claude-opus-4-6",
            "claude-3-5-haiku-20241022",
            "claude-3-haiku-20240307",
        ]
        for model_id in preferred:
            if model_id in ids:
                return model_id
        for model_id in ids:
            if "sonnet" in model_id:
                return model_id
        if ids:
            return ids[0]
        raise RuntimeError("No Anthropic models are available for this API key")

    def generate_structured(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        tool_name = "emit_industry_extraction"
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2500,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
            tools=[
                {
                    "name": tool_name,
                    "description": "Return the structured industry analysis extraction JSON.",
                    "input_schema": schema,
                }
            ],
            tool_choice={"type": "tool", "name": tool_name},
        )
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == tool_name:
                return dict(block.input)
        text = self._text_from_response(response)
        return self._parse_json_object(text)

    def generate_text(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=3500,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        return self._text_from_response(response)

    @staticmethod
    def _text_from_response(response) -> str:
        parts = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "\n".join(parts).strip()

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
            raise
