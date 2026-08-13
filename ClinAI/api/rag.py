from __future__ import annotations

import math
from typing import Any

import chromadb


class PatientVectorStore:
    def __init__(self, persist_path: str) -> None:
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(
            name="clinai_patients",
            metadata={"hnsw:space": "cosine"},
        )

    def index_patient(self, patient: dict[str, Any]) -> None:
        patient_id = str(patient["patient_id"])
        document = self._document(patient)
        metadata = {
            "patient_id": patient_id,
            "name": str(patient.get("name") or "NA"),
            "age": str(patient.get("age") or "NA"),
            "gender": str(patient.get("gender") or "NA"),
        }
        self.collection.upsert(
            ids=[patient_id],
            documents=[document],
            metadatas=[metadata],
        )

    def search(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        count = self.collection.count()
        if count == 0:
            return []
        result = self.collection.query(
            query_texts=[query],
            n_results=min(limit, count),
            include=["distances", "metadatas", "documents"],
        )
        rows: list[dict[str, Any]] = []
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        documents = result.get("documents", [[]])[0]
        for patient_id, distance, metadata, document in zip(
            ids, distances, metadatas, documents
        ):
            similarity = max(0.0, 1.0 - float(distance))
            score = int(math.ceil(similarity * 100))
            rows.append(
                {
                    "patient_id": patient_id,
                    "distance": float(distance),
                    "relevance_score": score,
                    "metadata": metadata or {},
                    "document": document,
                }
            )
        return rows

    @staticmethod
    def _document(patient: dict[str, Any]) -> str:
        return "\n".join(
            [
                f"Patient: {patient.get('name', 'NA')}",
                f"Age: {patient.get('age', 'NA')}",
                f"Gender: {patient.get('gender', 'NA')}",
                f"Summary: {patient.get('summary', '')}",
                f"Keywords: {patient.get('keywords', '')}",
                f"Prescriptions: {patient.get('prescriptions', '')}",
                f"Timeline: {patient.get('timeline', '')}",
                f"Clinical note: {patient.get('note', '')}",
            ]
        )
