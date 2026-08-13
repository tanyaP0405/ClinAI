from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


API_DIR = Path(__file__).resolve().parent
PROJECT_DIR = API_DIR.parent
load_dotenv(API_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    database_path: Path = Path(
        os.getenv("DATABASE_PATH", str(PROJECT_DIR / "data" / "clinai.db"))
    )
    chroma_path: Path = Path(
        os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_DIR / "data" / "chroma"))
    )


settings = Settings()
