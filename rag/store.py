"""Индекс FAISS и карточки фрагментов на диске.

Векторы нормированы, поиск идёт по скалярному произведению — это косинусная схожесть.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from rag.embeddings import Embedder, load_embedder
from rag.errors import RagError
from rag.models import Chunk, Hit

CHUNKS_FILE = "chunks.json"
INDEX_FILE = "faiss.index"


class IndexStore:
    def __init__(self, embedder: Embedder, index: faiss.Index, chunks: list[Chunk]) -> None:
        self.embedder = embedder
        self.index = index
        self.chunks = chunks

    @classmethod
    def build(cls, embedder: Embedder, vectors: np.ndarray, chunks: list[Chunk]) -> IndexStore:
        if len(chunks) == 0:
            raise RagError("Нечего индексировать: фрагментов нет")
        if vectors.shape[0] != len(chunks):
            raise RagError("Число векторов не совпало с числом фрагментов")
        matrix = np.ascontiguousarray(vectors, dtype=np.float32)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        return cls(embedder, index, chunks)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / INDEX_FILE))
        payload = [chunk.to_dict() for chunk in self.chunks]
        (directory / CHUNKS_FILE).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.embedder.save(directory)

    @classmethod
    def load(cls, directory: Path) -> IndexStore:
        index_path = directory / INDEX_FILE
        chunks_path = directory / CHUNKS_FILE
        if not index_path.exists() or not chunks_path.exists():
            raise RagError(
                "Индекс не найден. Сначала выполните: python -m rag ingest <папка>"
            )
        index = faiss.read_index(str(index_path))
        raw_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        chunks = [Chunk.from_dict(item) for item in raw_chunks]
        embedder = load_embedder(directory)
        return cls(embedder, index, chunks)

    def search(self, query: str, top_k: int) -> list[Hit]:
        if self.index.ntotal == 0 or not self.chunks:
            return []
        vector = self.embedder.embed([query])
        k = min(top_k, self.index.ntotal)
        scores, ids = self.index.search(np.ascontiguousarray(vector, dtype=np.float32), k)
        hits: list[Hit] = []
        for score, idx in zip(scores[0], ids[0]):
            if int(idx) < 0:
                continue
            hits.append(Hit(chunk=self.chunks[int(idx)], score=float(score)))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits
