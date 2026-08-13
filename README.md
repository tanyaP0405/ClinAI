# ClinAI — Agentic Healthcare Virtual Assistant

ClinAI converts doctor–patient conversations into structured clinical records,
preserves an editable patient timeline, and lets healthcare staff retrieve
historical context through natural-language vector search or a tool-calling
assistant.

## What it does

- **Voice clinical capture** — browser recording and Groq Whisper transcription.
- **Structured extraction** — one cost-efficient Groq call extracts demographics,
  summary, chronology, keywords, and prescriptions.
- **Complete patient workflow** — browse, create, retrieve, and update records;
  edit summaries, prescriptions, keywords, and the visual event timeline.
- **Agentic RAG** — Chroma vector retrieval across summaries, timelines,
  prescriptions, keywords, and notes.
- **Groq function-calling agent** — chooses between patient search, record
  retrieval, recent-record listing, and structured record creation.
- **Session memory** — multi-turn conversations are retained in SQLite.
- **Streaming UX** — the agent endpoint streams status, tool traces, and response
  tokens over Server-Sent Events.
- **Enterprise voice workspace** — a React and TypeScript capture flow provides
  recording controls, Groq Whisper transcription, and human review before save.

## Architecture

```text
Original clinical UI + React/TypeScript voice workspace
                         |
                      FastAPI
             ____________|____________
            |            |            |
        Groq LLM     SQLite       Chroma
     + Whisper STT   records      vector RAG
     + tool calling  + memory
```

MCP is intentionally not part of the runtime architecture. ClinAI is a single
application with a controlled set of local tools, so Groq's OpenAI-compatible
function calling provides a simpler and more direct agent loop.

## Stack

- React, TypeScript, Vite, and Ant Design
- HTML, Bootstrap, and JavaScript for existing clinical pages
- FastAPI, Python
- Groq Llama 3.3 70B and Whisper Large v3
- Chroma vector database
- SQLite
- Server-Sent Events

## Run locally

Requirements: Python 3.10+ and Node.js 18+.

1. Configure the API:

   ```powershell
   Copy-Item ClinAI\api\.env.example ClinAI\api\.env
   ```

   Add your free Groq key to `ClinAI/api/.env`:

   ```env
   GROQ_API_KEY=gsk_your_key
   GROQ_MODEL=llama-3.3-70b-versatile
   ```

2. Install and build:

   ```powershell
   cd ClinAI
   python -m pip install -e .
   cd frontend-react
   npm install
   npm run build
   ```

3. Start FastAPI:

   ```powershell
   cd ..\api
   python -m uvicorn main:app --reload
   ```

4. Open:

   - Clinical workflow: <http://127.0.0.1:8000>
   - React voice capture workspace: <http://127.0.0.1:8000/create>
   - Legacy capture fallback: <http://127.0.0.1:8000/legacy/create>
   - API documentation: <http://127.0.0.1:8000/docs>

## API highlights

- `POST /transcribe` — Groq Whisper transcription
- `POST /save_record` — structured extraction and persistence
- `POST /api/search` — Chroma semantic retrieval
- `POST /api/chat` — function-calling agent
- `POST /api/chat/stream` — streaming agent with tool traces
- `PATCH /patient/{id}/{field}` — update summary, timeline, prescriptions, or keywords

## Security note

Clinical data remains local in SQLite and Chroma. The configured Groq endpoints
receive transcription or LLM inputs. This portfolio project is not a certified
medical device and should not be used with real protected health information
without an appropriate compliance review.
