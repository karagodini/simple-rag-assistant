"""Структуры данных пайплайна. Никакой логики, только поля."""

from __future__ import annotations

from dataclasses import asdict, dataclass


def format_citation(
    source: str,
    fragment: int,
    *,
    section: str | None = None,
    page: int | None = None,
) -> str:
    """Ссылка вида [файл, фрагмент N], с разделом или страницей, если они есть."""
    parts = [source]
    if section:
        parts.append(f"раздел «{section}»")
    if page is not None:
        parts.append(f"стр. {page}")
    parts.append(f"фрагмент {fragment}")
    return "[" + ", ".join(parts) + "]"


@dataclass
class RawDocument:
    """Один исходный текст до нарезки: файл, страница PDF или весь txt/md."""

    text: str
    source: str
    page: int | None = None
    kind: str = "txt"


@dataclass
class Chunk:
    id: int
    text: str
    source: str
    page: int | None = None
    section: str | None = None

    def citation(self) -> str:
        return format_citation(
            self.source, self.id, section=self.section, page=self.page
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Chunk:
        return cls(
            id=int(data["id"]),
            text=str(data["text"]),
            source=str(data["source"]),
            page=data.get("page"),
            section=data.get("section"),
        )


@dataclass
class Hit:
    chunk: Chunk
    score: float


@dataclass
class Source:
    source: str
    fragment: int
    page: int | None
    section: str | None
    score: float
    text: str

    def citation(self) -> str:
        return format_citation(
            self.source, self.fragment, section=self.section, page=self.page
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["citation"] = self.citation()
        return data


@dataclass
class Answer:
    answer: str
    found: bool
    mode: str
    sources: list[Source]

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "found": self.found,
            "mode": self.mode,
            "sources": [source.to_dict() for source in self.sources],
        }
