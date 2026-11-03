from rag.chunking import build_chunks, chunk_text, split_markdown_sections
from rag.models import RawDocument


def test_overlap_is_shared_by_neighbors():
    text = "абвгдежзик" * 30
    chunks = chunk_text(text, chunk_size=100, overlap=25)
    assert len(chunks) >= 2
    assert chunks[0][-25:] == chunks[1][:25]
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_blank_lines_keep_short_paragraphs_apart():
    text = "Лендинг стоит 80000 рублей.\n\nЧат-бот стоит 120000 рублей."
    assert chunk_text(text, chunk_size=800, overlap=100) == [
        "Лендинг стоит 80000 рублей.",
        "Чат-бот стоит 120000 рублей.",
    ]


def test_short_text_is_one_chunk():
    assert chunk_text("  короткий текст  ", chunk_size=100, overlap=10) == ["короткий текст"]


def test_empty_text_returns_no_chunks():
    assert chunk_text("   \n\t") == []


def test_invalid_overlap_raises():
    try:
        chunk_text("текст", chunk_size=10, overlap=10)
    except ValueError:
        return
    raise AssertionError("overlap == chunk_size должен быть ошибкой")


def test_markdown_sections_keep_title_and_metadata():
    body = "Заявление подаётся за 14 дней. " * 40
    document = RawDocument(
        text=f"# Регламент\n\nВступление про офис.\n\n## Отпуск\n\n{body}",
        source="reglament.md",
        kind="md",
    )
    sections = split_markdown_sections(document.text)
    titles = [title for title, _ in sections]
    assert "Отпуск" in titles

    chunks = build_chunks([document], chunk_size=120, overlap=20)
    vacation = [chunk for chunk in chunks if chunk.section == "Отпуск"]
    assert len(vacation) >= 2
    assert vacation[0].source == "reglament.md"
    assert vacation[0].page is None
    assert "Отпуск" in vacation[0].text
    assert vacation[0].citation().startswith("[reglament.md, раздел «Отпуск», фрагмент ")


def test_pdf_page_is_kept_on_chunks():
    document = RawDocument(
        text="Страница про отпуск на 28 дней.",
        source="rules.pdf",
        page=2,
        kind="pdf",
    )
    chunks = build_chunks([document], chunk_size=800, overlap=100)
    assert len(chunks) == 1
    assert chunks[0].page == 2
    assert chunks[0].source == "rules.pdf"
    assert "стр. 2" in chunks[0].citation()
