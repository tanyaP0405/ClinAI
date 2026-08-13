"""Check the services required by the current ClinAI architecture."""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / "ClinAI" / "api" / ".env"
load_dotenv(ENV_PATH)


def check(label: str, action) -> bool:
    try:
        detail = action()
        print(f"OK    {label} — {detail}")
        return True
    except Exception as exc:
        print(f"FAIL  {label} — {str(exc)[:180]}")
        return False


def groq_check() -> str:
    from groq import Groq

    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError(f"GROQ_API_KEY is missing from {ENV_PATH}")
    response = Groq(api_key=key).chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=[{"role": "user", "content": "Reply with exactly OK"}],
        max_tokens=5,
        temperature=0,
    )
    return (response.choices[0].message.content or "").strip()


def sqlite_check() -> str:
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE health (status TEXT)")
    connection.execute("INSERT INTO health VALUES ('ready')")
    status = connection.execute("SELECT status FROM health").fetchone()[0]
    connection.close()
    return status


def chroma_check() -> str:
    import chromadb

    client = chromadb.Client()
    collection = client.get_or_create_collection("clinai_health")
    collection.upsert(ids=["health"], documents=["clinical vector search ready"])
    result = collection.query(query_texts=["clinical search"], n_results=1)
    return "vector query ready" if result["ids"][0] else "no result"


if __name__ == "__main__":
    print(f"ClinAI service check\nEnvironment: {ENV_PATH}\n")
    checks = [
        check("Groq LLM", groq_check),
        check("SQLite", sqlite_check),
        check("Chroma", chroma_check),
    ]
    print(f"\nResult: {sum(checks)}/{len(checks)} checks passed")
    sys.exit(0 if all(checks) else 1)
