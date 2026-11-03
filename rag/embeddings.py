"""Эмбеддинги: TF-IDF без сети или опционально sentence-transformers.

TF-IDF считает похожесть по словам. Лёгкий стеммер срезает частые русские
окончания, чтобы «отпуска» и «отпуск» попали в один признак. Это не понимание
смысла: перефразированный вопрос без общих слов найдётся только моделью sbert.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from rag.errors import RagError

_TOKEN = re.compile(r"[0-9a-zа-я]{2,}", re.IGNORECASE)
# Длинные окончания раньше коротких, иначе «ами» никогда не сработает.
_ENDINGS = (
    "иями",
    "ями",
    "ами",
    "ого",
    "ему",
    "ому",
    "ыми",
    "ими",
    "ах",
    "ях",
    "ов",
    "ев",
    "ом",
    "ем",
    "ой",
    "ий",
    "ый",
    "ых",
    "их",
    "ая",
    "яя",
    "ое",
    "ее",
    "ые",
    "ие",
    "ию",
    "ью",
    "ия",
    "ья",
    "а",
    "я",
    "у",
    "ю",
    "ы",
    "и",
    "е",
    "о",
    "й",
)


def stem(token: str) -> str:
    """Срезать одно типичное окончание, если основа останется не короче 4 букв."""
    if token.isdigit() or len(token) <= 4:
        return token
    for ending in _ENDINGS:
        if token.endswith(ending) and len(token) - len(ending) >= 4:
            return token[: -len(ending)]
    return token


def tokenize(text: str) -> list[str]:
    """Анализатор для TfidfVectorizer. Функция именованная: её можно сохранить joblib."""
    normalized = text.lower().replace("ё", "е")
    return [stem(token) for token in _TOKEN.findall(normalized)]


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    matrix = np.ascontiguousarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.divide(matrix, norms, out=matrix, where=norms > 0)
    return matrix


class Embedder(Protocol):
    kind: str

    def fit(self, texts: list[str]) -> None: ...

    def embed(self, texts: list[str]) -> np.ndarray: ...

    def save(self, directory: Path) -> None: ...


class TfidfEmbedder:
    kind = "tfidf"

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(analyzer=tokenize, norm="l2", sublinear_tf=True)
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        if not texts:
            raise RagError("Нет текста для обучения TF-IDF")
        self.vectorizer.fit(texts)
        self._fitted = True

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RagError("TF-IDF ещё не обучен")
        matrix = self.vectorizer.transform(texts)
        return l2_normalize(matrix.toarray())

    def save(self, directory: Path) -> None:
        import joblib

        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, directory / "vectorizer.joblib")
        (directory / "embedder.json").write_text(
            json.dumps({"type": self.kind}, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> TfidfEmbedder:
        import joblib

        embedder = cls()
        embedder.vectorizer = joblib.load(directory / "vectorizer.joblib")
        embedder._fitted = True
        return embedder


class SbertEmbedder:
    """Мультиязычные эмбеддинги. Пакет sentence-transformers ставится отдельно."""

    kind = "sbert"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RagError(
                "Для режима sbert установите зависимости: "
                "pip install -r requirements-optional.txt"
            ) from exc
        self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, texts: list[str]) -> None:
        # Предобученная модель не учится на корпусе, но проверяем, что она открывается.
        self._load_model()

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._load_model()
        vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return l2_normalize(np.asarray(vectors))

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        payload = {"type": self.kind, "model": self.model_name}
        (directory / "embedder.json").write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> SbertEmbedder:
        payload = json.loads((directory / "embedder.json").read_text(encoding="utf-8"))
        return cls(payload["model"])


def create_embedder(kind: str, sbert_model: str) -> Embedder:
    if kind == "tfidf":
        return TfidfEmbedder()
    if kind == "sbert":
        return SbertEmbedder(sbert_model)
    raise RagError("Неизвестный эмбеддер. Используйте tfidf или sbert")


def load_embedder(directory: Path) -> Embedder:
    description_path = directory / "embedder.json"
    if not description_path.exists():
        raise RagError(f"В индексе нет embedder.json: {directory}")
    payload = json.loads(description_path.read_text(encoding="utf-8"))
    kind = payload.get("type")
    if kind == "tfidf":
        return TfidfEmbedder.load(directory)
    if kind == "sbert":
        return SbertEmbedder.load(directory)
    raise RagError(f"Неизвестный тип эмбеддера в индексе: {kind}")
