"""Minimal Gemini Flash model test."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / "ClinAI" / "api" / ".env")
key = os.getenv("GEMINI_API_KEY", "").strip()

print("Testing Gemini key (Flash models only, minimal tokens)...")

models = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
    "gemini-1.5-flash",
]
for model in models:
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)
        resp = client.models.generate_content(
            model=model,
            contents="OK",
            config=types.GenerateContentConfig(max_output_tokens=3, temperature=0),
        )
        text = (resp.text or "").strip()
        print(f"OK   {model}: {text[:40]}")
    except Exception as exc:
        err = str(exc).replace("\n", " ")[:140]
        print(f"FAIL {model}: {err}")
