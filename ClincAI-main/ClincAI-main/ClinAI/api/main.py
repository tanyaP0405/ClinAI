from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent import ClinicalAgent
from config import settings
from database import PatientRepository
from llm import GroqService
from rag import PatientVectorStore


repository = PatientRepository(settings.database_path)
vectors: PatientVectorStore | None = None
llm: GroqService | None = None
agent: ClinicalAgent | None = None


class RecordRequest(BaseModel):
    idx: str = Field(min_length=1)
    conversation: str = ""
    notes: str = ""


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=10)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None


class FieldUpdate(BaseModel):
    value: str


@asynccontextmanager
async def lifespan(_: FastAPI):
    global vectors, llm, agent
    repository.initialize()
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    vectors = PatientVectorStore(str(settings.chroma_path))
    llm = GroqService()
    agent = ClinicalAgent(llm, repository, vectors)

    for patient in repository.list_patients(limit=1000):
        full = repository.get_patient(patient["patient_id"])
        if full:
            vectors.index_patient(full)
    yield


app = FastAPI(
    title="ClinAI Agentic Healthcare API",
    description="Groq tool calling, SQLite memory, and Chroma Agentic RAG.",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_services() -> tuple[GroqService, PatientVectorStore, ClinicalAgent]:
    if llm is None or vectors is None or agent is None:
        raise HTTPException(status_code=503, detail="Services are starting")
    return llm, vectors, agent


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "llm": settings.groq_model,
        "database": "sqlite",
        "vector_store": "chroma",
        "patients_indexed": vectors.collection.count() if vectors else 0,
    }


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)) -> JSONResponse:
    if not settings.groq_api_key:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not configured")
    audio_bytes = await file.read()

    def request_transcription() -> requests.Response:
        return requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {settings.groq_api_key}"},
            files={"file": (file.filename, audio_bytes, file.content_type)},
            data={"model": "whisper-large-v3"},
            timeout=90,
        )

    response = await asyncio.to_thread(request_transcription)
    if not response.ok:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Groq transcription failed: {response.text[:300]}",
        )
    return JSONResponse({"transcription": response.json().get("text", "")})


@app.post("/label_conversation")
async def label_conversation(request: Request) -> dict[str, str]:
    body = await request.json()
    full = "\n".join(
        part.strip()
        for part in [body.get("previous", ""), body.get("conversation", "")]
        if part.strip()
    )
    if not full:
        raise HTTPException(status_code=400, detail="Conversation is required")
    groq, _, _ = require_services()
    labeled = await asyncio.to_thread(groq.label_conversation, full)
    return {"labeled_conversation": labeled}


@app.post("/save_record")
async def save_record(payload: RecordRequest) -> dict[str, Any]:
    if not payload.conversation.strip() and not payload.notes.strip():
        raise HTTPException(
            status_code=400, detail="Conversation or clinical notes are required"
        )
    _, _, clinical_agent = require_services()
    patient = await asyncio.to_thread(
        clinical_agent.create_patient_record,
        payload.idx.strip(),
        payload.conversation.strip(),
        payload.notes.strip(),
    )
    if "error" in patient:
        raise HTTPException(status_code=400, detail=patient["error"])
    return {
        "message": f"Record saved successfully for patient {payload.idx}",
        "patient": patient,
    }


@app.get("/api/patients")
def list_patients(limit: int = 100) -> list[dict[str, Any]]:
    return repository.list_patients(limit=min(max(limit, 1), 200))


@app.get("/api/patient/{patient_id}")
def get_patient(patient_id: str) -> dict[str, Any]:
    patient = repository.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@app.get("/patient/{patient_id}/details")
def get_patient_details(patient_id: str) -> dict[str, Any]:
    patient = repository.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return {
        "summary": patient["summary"],
        "keywords": [
            keyword.strip()
            for keyword in patient["keywords"].split(",")
            if keyword.strip()
        ],
        "name": patient["name"],
        "age": patient["age"],
        "gender": patient["gender"],
    }


@app.patch("/patient/{patient_id}/{field}")
def update_patient_field(
    patient_id: str, field: str, payload: dict[str, Any]
) -> dict[str, str]:
    if field not in {"summary", "timeline", "prescriptions", "keywords"}:
        raise HTTPException(status_code=404, detail="Unsupported patient field")
    value = payload.get(field, payload.get("value"))
    if not isinstance(value, str):
        raise HTTPException(status_code=400, detail=f"{field} must be a string")
    if not repository.update_field(patient_id, field, value):
        raise HTTPException(status_code=404, detail="Patient not found")
    patient = repository.get_patient(patient_id)
    if vectors and patient:
        vectors.index_patient(patient)
    return {"message": f"{field.capitalize()} updated successfully"}


@app.post("/api/search")
async def semantic_search(payload: SearchRequest) -> dict[str, Any]:
    _, vector_store, _ = require_services()
    matches = await asyncio.to_thread(
        vector_store.search, payload.query.strip(), payload.limit
    )
    results: list[dict[str, Any]] = []
    for match in matches:
        patient = repository.get_patient(match["patient_id"])
        if patient:
            patient["relevance_score"] = match["relevance_score"]
            patient["relevance_reason"] = "Chroma semantic similarity"
            results.append(patient)
    return {
        "results": results,
        "total_found": len(results),
        "query": payload.query,
        "retrieval": "Chroma cosine vector search",
    }


@app.post("/api/chat")
async def chat(payload: ChatRequest) -> dict[str, Any]:
    _, _, clinical_agent = require_services()
    session_id = payload.session_id or str(uuid.uuid4())
    return await asyncio.to_thread(
        clinical_agent.respond, session_id, payload.message.strip()
    )


@app.post("/api/chat/stream")
async def chat_stream(payload: ChatRequest) -> StreamingResponse:
    _, _, clinical_agent = require_services()
    session_id = payload.session_id or str(uuid.uuid4())

    async def events():
        yield f"data: {json.dumps({'type': 'status', 'message': 'Planning tool use...'})}\n\n"
        result = await asyncio.to_thread(
            clinical_agent.respond, session_id, payload.message.strip()
        )
        for trace in result["tool_trace"]:
            yield f"data: {json.dumps({'type': 'tool', **trace})}\n\n"
        words = result["message"].split(" ")
        for word in words:
            yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"
            await asyncio.sleep(0.015)
        yield f"data: {json.dumps({'type': 'done', 'session_id': session_id})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


legacy_frontend = Path(__file__).resolve().parents[1] / "frontend"
create_frontend = Path(__file__).resolve().parents[1] / "frontend-react" / "dist"
app.mount("/static", StaticFiles(directory=legacy_frontend / "static"), name="static")
if create_frontend.exists():
    app.mount(
        "/create-assets",
        StaticFiles(directory=create_frontend),
        name="create-assets",
    )


@app.get("/", include_in_schema=False)
async def home_page():
    return FileResponse(legacy_frontend / "index.html")


@app.get("/create", include_in_schema=False)
async def create_page():
    if create_frontend.exists():
        return FileResponse(create_frontend / "index.html")
    return FileResponse(legacy_frontend / "create.html")


@app.get("/legacy/create", include_in_schema=False)
async def legacy_create_page():
    return FileResponse(legacy_frontend / "create.html")


@app.get("/patients", include_in_schema=False)
async def patients_page():
    return FileResponse(legacy_frontend / "patients.html")


@app.get("/patient/{patient_id}", include_in_schema=False)
async def patient_page(patient_id: str):
    return FileResponse(legacy_frontend / "patient.html")


@app.get("/update-patients", include_in_schema=False)
async def update_patients_page():
    return FileResponse(legacy_frontend / "updatepatients.html")


@app.get("/update-patient/{patient_id}", include_in_schema=False)
async def update_patient_page(patient_id: str):
    return FileResponse(legacy_frontend / "updatepatient.html")


@app.get("/semantic-search", include_in_schema=False)
async def semantic_search_page():
    return FileResponse(legacy_frontend / "semantic-search.html")
