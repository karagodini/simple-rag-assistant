from pathlib import Path

import pytest

from rag.config import Settings
from rag.loaders import load_documents
from rag.pipeline import RagPipeline
from rag.store import IndexStore


def _write_corpus(root: Path) -> None:
    docs = root / "docs"
    docs.mkdir()
    (docs / "reglament.md").write_text(
        "# Правила\n\n## Отпуск\n\n"
        "Ежегодный оплачиваемый отпуск составляет 28 календарных дней. "
        "Заявление на отпуск подаётся за 14 календарных дней.\n",
        encoding="utf-8",
    )
    (docs / "price.txt").write_text(
        "Чат-бот с ответами по документам стоит 120000 рублей. "
        "Срок выполнения 15 рабочих дней.\n",
        encoding="utf-8",
    )


def _pdf_with_text(path: Path, text: str) -> None:
    """Минимальный PDF с одной строкой латиницы. Кириллицу Helvetica не умеет."""
    safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 50 750 Td ({safe}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
        ),
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output += f"{number} 0 obj\n".encode("ascii")
        output += obj + b"\nendobj\n"
    xref = len(output)
    output += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    output += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        output += f"{offset:010d} 00000 n \n".encode("ascii")
    output += (
        f"trailer<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n"
    ).encode("ascii")
    path.write_bytes(output)


def test_search_returns_the_document_that_contains_the_fact(tmp_path: Path):
    _write_corpus(tmp_path)
    settings = Settings(index_dir=tmp_path / "index", openai_api_key=None, min_score=0.05)
    pipeline = RagPipeline(settings)
    report = pipeline.ingest([tmp_path / "docs"])
    assert report["chunks"] >= 2

    store = IndexStore.load(tmp_path / "index")
    hits = store.search("за сколько дней подаётся заявление на отпуск", top_k=3)
    assert hits[0].chunk.source == "reglament.md"
    assert hits[0].chunk.section == "Отпуск"
    assert "28" in hits[0].chunk.text or "14" in hits[0].chunk.text
    assert hits[0].score > hits[-1].score or len(hits) == 1

    price_hits = store.search("сколько стоит чат-бот по документам", top_k=2)
    assert price_hits[0].chunk.source == "price.txt"
    assert "120000" in price_hits[0].chunk.text


def test_index_roundtrip_keeps_ranking(tmp_path: Path):
    _write_corpus(tmp_path)
    settings = Settings(index_dir=tmp_path / "index", openai_api_key=None)
    RagPipeline(settings).ingest([tmp_path / "docs"])

    first = IndexStore.load(tmp_path / "index")
    second = IndexStore.load(tmp_path / "index")
    query = "оплачиваемый отпуск 28 календарных дней"
    assert [hit.chunk.id for hit in first.search(query, 2)] == [
        hit.chunk.id for hit in second.search(query, 2)
    ]
    assert first.search(query, 1)[0].score == pytest.approx(
        second.search(query, 1)[0].score
    )


def test_pdf_loader_keeps_page_number(tmp_path: Path):
    pdf_path = tmp_path / "policy.pdf"
    _pdf_with_text(pdf_path, "Vacation request must be filed 14 days ahead.")
    documents = load_documents([pdf_path])
    assert len(documents) == 1
    assert documents[0].page == 1
    assert documents[0].source == "policy.pdf"
    assert "14 days" in documents[0].text

    settings = Settings(index_dir=tmp_path / "index", openai_api_key=None, min_score=0.05)
    pipeline = RagPipeline(settings)
    pipeline.ingest([pdf_path])
    answer = pipeline.ask("How many days ahead is the vacation request filed?")
    assert answer.found
    assert answer.sources[0].page == 1
    assert "policy.pdf" in answer.answer
    assert "стр. 1" in answer.answer
