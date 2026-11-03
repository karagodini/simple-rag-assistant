"""Нарезка текста на фрагменты фиксированной длины с перекрытием.

Перекрытие нужно, чтобы фраза на стыке двух кусков не потерялась:
следующий фрагмент начинается на `overlap` символов раньше конца предыдущего.
"""

from __future__ import annotations

import re

from rag.models import Chunk, RawDocument

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def _sliding_window(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Окно фиксированной длины. Соседи делят хвост длиной overlap символов."""
    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        piece = text[start : start + chunk_size].strip()
        if piece and not (chunks and piece in chunks[-1]):
            chunks.append(piece)
        if start + chunk_size >= length:
            break
        start += step
    return chunks


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """Нарезать текст.

    Абзацы, разделённые пустой строкой, не склеиваются: так одна цена или одно
    правило не смешивается с соседним. Длинный абзац режется окном с перекрытием,
    чтобы фраза на стыке попала в оба фрагмента.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size должен быть больше нуля")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap должен быть >= 0 и меньше chunk_size")

    cleaned = text.strip()
    if not cleaned:
        return []

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", cleaned) if part.strip()]
    chunks: list[str] = []
    for paragraph in paragraphs:
        chunks.extend(_sliding_window(paragraph, chunk_size, overlap))
    return chunks


def split_markdown_sections(text: str) -> list[tuple[str | None, str]]:
    """Разбить markdown по заголовкам. Текст секции включает сам заголовок."""
    matches = list(_HEADING.finditer(text))
    if not matches:
        stripped = text.strip()
        return [(None, stripped)] if stripped else []

    sections: list[tuple[str | None, str]] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append((None, preamble))

    for index, match in enumerate(matches):
        title = match.group(2).strip()
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        if not body:
            continue
        sections.append((title, f"{title}\n{body}".strip()))
    return sections


def build_chunks(
    documents: list[RawDocument],
    *,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    """Нарезать документы и проставить файл, страницу и раздел."""
    chunks: list[Chunk] = []
    next_id = 1
    for document in documents:
        if document.kind == "md":
            sections = split_markdown_sections(document.text)
        else:
            stripped = document.text.strip()
            sections = [(None, stripped)] if stripped else []

        for section, text in sections:
            for piece in chunk_text(text, chunk_size, overlap):
                chunks.append(
                    Chunk(
                        id=next_id,
                        text=piece,
                        source=document.source,
                        page=document.page,
                        section=section,
                    )
                )
                next_id += 1
    return chunks
