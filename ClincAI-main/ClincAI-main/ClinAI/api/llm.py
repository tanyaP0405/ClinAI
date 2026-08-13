from __future__ import annotations

import json
import re
from typing import Any

from groq import Groq

from config import settings


class GroqService:
    def __init__(self) -> None:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not configured in ClinAI/api/.env")
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.groq_model

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 1400,
        response_format: dict[str, str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"
        return self.client.chat.completions.create(**kwargs)

    def extract_record(self, conversation: str, note: str) -> dict[str, Any]:
        prompt = f"""
Extract a structured clinical record from the provided doctor-patient material.
Use only facts present in the input. Never invent clinical details.

Return one JSON object with exactly these fields:
- name: patient name or "NA"
- age: age as a string or "NA"
- gender: gender or "NA"
- summary: concise clinical summary, maximum four sentences
- timeline: array of chronological clinical event strings
- keywords: comma-separated conditions, symptoms, medications, and tests
- prescriptions: newline-separated entries formatted exactly as
  "Drug: <name>, Dose: <dose>, Route: <route>, Status: <active|stopped|continuing>"
  Use "NA" for missing values, or "No prescriptions found." when none exist.

Clinical note:
{note}

Conversation:
{conversation}
""".strip()
        response = self.complete(
            [
                {
                    "role": "system",
                    "content": "You are a precise clinical documentation assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=1800,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        parsed = self._parse_json(content)
        timeline = parsed.get("timeline", [])
        if not isinstance(timeline, list):
            timeline = [str(timeline)] if timeline else []
        return {
            "name": str(parsed.get("name") or "NA"),
            "age": str(parsed.get("age") or "NA"),
            "gender": str(parsed.get("gender") or "NA"),
            "summary": str(parsed.get("summary") or ""),
            "timeline": [str(event) for event in timeline],
            "keywords": str(parsed.get("keywords") or ""),
            "prescriptions": str(
                parsed.get("prescriptions") or "No prescriptions found."
            ),
        }

    def label_conversation(self, conversation: str) -> str:
        if "Doctor:" in conversation and "Patient:" in conversation:
            return conversation.strip()
        response = self.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a strict transcript formatter. Prefix each existing "
                        "speaker turn with Doctor: or Patient:. The doctor speaks "
                        "first. Preserve every original word exactly. Do not answer, "
                        "continue, paraphrase, correct, or add dialogue. Return only "
                        "the labeled version of the supplied transcript."
                    ),
                },
                {
                    "role": "user",
                    "content": f"FORMAT THIS TRANSCRIPT ONLY:\n{conversation}",
                },
            ],
            temperature=0,
            max_tokens=1600,
        )
        return (response.choices[0].message.content or conversation).strip()

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if not match:
                raise ValueError("Groq returned invalid structured clinical data")
            return json.loads(match.group(0))
