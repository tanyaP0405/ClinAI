from __future__ import annotations

import json
from typing import Any, Callable

from database import PatientRepository
from llm import GroqService
from rag import PatientVectorStore


SYSTEM_PROMPT = """
You are ClinAI, an enterprise healthcare workflow assistant.
Use tools whenever the user asks about stored patients or requests an action.
Never fabricate patient data. State clearly when no matching record exists.
You may create a clinical record only when a patient ID and clinical material
are provided. Keep responses concise, professional, and grounded in tool output.
This is a documentation assistant, not a substitute for medical judgment.
""".strip()


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_patients",
            "description": (
                "Agentic RAG search over summaries, timelines, prescriptions, "
                "keywords, and notes using clinical natural language."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 8},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_patient",
            "description": "Retrieve the complete stored record for one patient ID.",
            "parameters": {
                "type": "object",
                "properties": {"patient_id": {"type": "string"}},
                "required": ["patient_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_patients",
            "description": "List recently updated patient records.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20}
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_patient_record",
            "description": (
                "Extract and save a structured clinical record from a conversation "
                "or clinical note."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "conversation": {"type": "string"},
                    "note": {"type": "string"},
                },
                "required": ["patient_id"],
                "additionalProperties": False,
            },
        },
    },
]


class ClinicalAgent:
    def __init__(
        self,
        llm: GroqService,
        repository: PatientRepository,
        vectors: PatientVectorStore,
    ) -> None:
        self.llm = llm
        self.repository = repository
        self.vectors = vectors
        self.tool_map: dict[str, Callable[..., Any]] = {
            "search_patients": self.search_patients,
            "get_patient": self.get_patient,
            "list_patients": self.list_patients,
            "create_patient_record": self.create_patient_record,
        }

    def respond(self, session_id: str, user_message: str) -> dict[str, Any]:
        self.repository.add_message(session_id, "user", user_message)
        history = self.repository.get_messages(session_id, limit=16)
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(
            {"role": item["role"], "content": item["content"]}
            for item in history
            if item["role"] in {"user", "assistant"}
        )
        tool_trace: list[dict[str, Any]] = []

        for _ in range(4):
            completion = self.llm.complete(
                messages,
                temperature=0.1,
                max_tokens=1200,
                tools=TOOLS,
                tool_choice="auto",
            )
            assistant = completion.choices[0].message
            tool_calls = assistant.tool_calls or []
            if not tool_calls:
                answer = (assistant.content or "I could not complete that request.").strip()
                self.repository.add_message(session_id, "assistant", answer)
                return {
                    "message": answer,
                    "session_id": session_id,
                    "tool_trace": tool_trace,
                }

            messages.append(assistant.model_dump(exclude_none=True))
            for call in tool_calls:
                name = call.function.name
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                    if name not in self.tool_map:
                        raise ValueError(f"Unknown tool: {name}")
                    result = self.tool_map[name](**arguments)
                    output = json.dumps(result, default=str)
                    tool_trace.append({"tool": name, "arguments": arguments})
                except Exception as exc:
                    output = json.dumps({"error": str(exc)})
                    tool_trace.append({"tool": name, "error": str(exc)})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": name,
                        "content": output,
                    }
                )

        answer = "I reached the tool-execution limit. Please narrow the request."
        self.repository.add_message(session_id, "assistant", answer)
        return {"message": answer, "session_id": session_id, "tool_trace": tool_trace}

    def search_patients(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        matches = self.vectors.search(query, limit=limit)
        patients: list[dict[str, Any]] = []
        for match in matches:
            patient = self.repository.get_patient(match["patient_id"])
            if patient:
                patient["relevance_score"] = match["relevance_score"]
                patients.append(patient)
        return patients

    def get_patient(self, patient_id: str) -> dict[str, Any]:
        return self.repository.get_patient(patient_id) or {
            "error": f"Patient {patient_id} was not found"
        }

    def list_patients(self, limit: int = 10) -> list[dict[str, Any]]:
        return self.repository.list_patients(limit=limit)

    def create_patient_record(
        self, patient_id: str, conversation: str = "", note: str = ""
    ) -> dict[str, Any]:
        if not conversation.strip() and not note.strip():
            return {"error": "A conversation or clinical note is required"}
        extracted = self.llm.extract_record(conversation, note)
        patient = self.repository.upsert_patient(
            {
                "patient_id": patient_id,
                "conversation": conversation,
                "note": note,
                **extracted,
            }
        )
        self.vectors.index_patient(patient)
        return patient
