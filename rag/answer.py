"""Ответ по найденным фрагментам: языковая модель или экстрактивный фолбэк."""

from __future__ import annotations

import re

from rag.config import NOT_FOUND, Settings
from rag.embeddings import tokenize
from rag.models import Answer, Chunk, Hit, Source

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def hits_to_sources(hits: list[Hit]) -> list[Source]:
    sources: list[Source] = []
    for hit in hits:
        chunk = hit.chunk
        sources.append(
            Source(
                source=chunk.source,
                fragment=chunk.id,
                page=chunk.page,
                section=chunk.section,
                score=round(hit.score, 4),
                text=chunk.text,
            )
        )
    return sources


def _shorten(text: str, limit: int = 500) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    cut = compact[:limit].rsplit(" ", 1)[0]
    return cut + "…"


def _body(chunk: Chunk) -> str:
    text = chunk.text.strip()
    if chunk.section and text.startswith(chunk.section):
        rest = text[len(chunk.section) :].strip()
        if rest:
            return rest
    return text


def select_excerpt(question: str, text: str, limit: int = 2) -> str:
    """Взять 1–2 предложения фрагмента, которые ближе всего к вопросу."""
    compact = " ".join(text.split())
    parts = [part for part in _SENTENCE.split(compact) if part]
    if len(parts) <= limit:
        return compact
    question_tokens = set(tokenize(question))
    scored = [(len(question_tokens & set(tokenize(part))), part) for part in parts]
    scored.sort(key=lambda item: item[0], reverse=True)
    if scored[0][0] == 0:
        return _shorten(compact)
    picked = {part for score, part in scored[:limit] if score > 0}
    return " ".join(part for part in parts if part in picked)


def extractive_answer(question: str, hits: list[Hit]) -> str:
    """Собрать ответ из цитат, без вызова модели."""
    lines = ["Ответ собран из найденных фрагментов:", ""]
    for hit in hits:
        quote = select_excerpt(question, _body(hit.chunk))
        lines.append(f"«{quote}» {hit.chunk.citation()}")
        lines.append("")
    return "\n".join(lines).strip()


def _build_context(hits: list[Hit]) -> str:
    blocks = []
    for hit in hits:
        blocks.append(f"Источник {hit.chunk.citation()}:\n{hit.chunk.text}")
    return "\n\n".join(blocks)


def generate_with_llm(question: str, hits: list[Hit], settings: Settings) -> str:
    """Сформулировать ответ через OpenAI-совместимый endpoint."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Пакет openai не установлен") from exc

    system = (
        "Ты ассистент по документам. Отвечай только по фрагментам из контекста. "
        "Каждый факт сопроводи ссылкой точно в том виде, в каком она указана "
        "у источника, например [reglament.md, фрагмент 2]. "
        f"Если во фрагментах нет ответа на вопрос, ответь ровно: {NOT_FOUND} "
        "Не используй общие знания и ничего не выдумывай. Пиши по-русски, кратко."
    )
    user = f"Контекст:\n{_build_context(hits)}\n\nВопрос: {question}"
    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    response = client.chat.completions.create(
        model=settings.openai_model,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        timeout=60,
    )
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("Модель вернула пустой ответ")
    return text


def compose_answer(question: str, hits: list[Hit], settings: Settings) -> Answer:
    """Решить, есть ли ответ в документах, и выбрать способ формулировки.

    `question` нужен и модели, и экстрактивному режиму: из фрагмента берутся
    одно-два предложения, ближе всего к формулировке вопроса.
    """
    relevant = [hit for hit in hits if hit.score >= settings.min_score]
    if not relevant:
        return Answer(answer=NOT_FOUND, found=False, mode="extractive", sources=[])

    if settings.llm_enabled():
        try:
            text = generate_with_llm(question, relevant, settings)
        except Exception:
            text = (
                "Внешняя модель недоступна, ответ собран из фрагментов.\n\n"
                + extractive_answer(question, relevant)
            )
            return Answer(
                answer=text,
                found=True,
                mode="extractive",
                sources=hits_to_sources(relevant),
            )
        found = NOT_FOUND not in text
        return Answer(
            answer=text,
            found=found,
            mode="llm",
            sources=hits_to_sources(relevant) if found else [],
        )

    return Answer(
        answer=extractive_answer(question, relevant),
        found=True,
        mode="extractive",
        sources=hits_to_sources(relevant),
    )
