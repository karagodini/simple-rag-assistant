"""Сквозной сценарий: загрузить файлы, построить индекс, ответить на вопрос."""

from __future__ import annotations

import threading
from pathlib import Path

from rag.answer import compose_answer
from rag.chunking import build_chunks
from rag.config import Settings
from rag.embeddings import create_embedder
from rag.errors import RagError
from rag.loaders import load_documents
from rag.models import Answer
from rag.store import IndexStore


class RagPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._store: IndexStore | None = None

    def ingest(self, paths: list[Path]) -> dict:
        documents = load_documents(paths)
        chunks = build_chunks(
            documents,
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        if not chunks:
            raise RagError("После нарезки не осталось фрагментов")

        embedder = create_embedder(self.settings.embedder, self.settings.sbert_model)
        texts = [chunk.text for chunk in chunks]
        embedder.fit(texts)
        vectors = embedder.embed(texts)
        store = IndexStore.build(embedder, vectors, chunks)
        with self._lock:
            store.save(self.settings.index_dir)
            self._store = store

        files = sorted({document.source for document in documents})
        return {
            "files": len(files),
            "file_names": files,
            "chunks": len(chunks),
            "index_dir": str(self.settings.index_dir),
            "embedder": self.settings.embedder,
        }

    def _get_store(self) -> IndexStore:
        with self._lock:
            if self._store is None:
                self._store = IndexStore.load(self.settings.index_dir)
            return self._store

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        cleaned = question.strip()
        if not cleaned:
            raise RagError("Пустой вопрос")
        store = self._get_store()
        hits = store.search(cleaned, top_k or self.settings.top_k)
        return compose_answer(cleaned, hits, self.settings)
