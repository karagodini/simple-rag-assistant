"""Настройки из окружения и значения по умолчанию."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

NOT_FOUND = "Не нашёл ответа в загруженных документах."


@dataclass
class Settings:
    index_dir: Path = Path(".rag_index")
    embedder: str = "tfidf"
    sbert_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4
    min_score: float = 0.25
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o-mini"

    def llm_enabled(self) -> bool:
        return bool(self.openai_api_key and self.openai_api_key.strip())


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def load_settings() -> Settings:
    """Прочитать .env (если файл есть) и переменные окружения."""
    load_dotenv()
    return Settings(
        index_dir=Path(os.getenv("RAG_INDEX_DIR", ".rag_index")),
        embedder=os.getenv("RAG_EMBEDDER", "tfidf").strip().lower() or "tfidf",
        sbert_model=os.getenv(
            "RAG_SBERT_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ).strip(),
        chunk_size=_env_int("RAG_CHUNK_SIZE", 800),
        chunk_overlap=_env_int("RAG_CHUNK_OVERLAP", 120),
        top_k=_env_int("RAG_TOP_K", 4),
        min_score=_env_float("RAG_MIN_SCORE", 0.25),
        openai_api_key=_blank_to_none(os.getenv("OPENAI_API_KEY")),
        openai_base_url=_blank_to_none(os.getenv("OPENAI_BASE_URL")),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini",
    )
