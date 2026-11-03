"""Чтение .md, .txt и .pdf. PDF разбирается постранично через pypdf."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from rag.errors import RagError
from rag.models import RawDocument

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


def _relative_source(path: Path, root: Path) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        relative = path.name
    return relative.as_posix()


def load_file(path: Path, root: Path) -> list[RawDocument]:
    suffix = path.suffix.lower()
    source = _relative_source(path, root)
    if suffix in {".md", ".txt"}:
        text = path.read_text(encoding="utf-8")
        return [RawDocument(text=text, source=source, kind="md" if suffix == ".md" else "txt")]
    if suffix == ".pdf":
        return _load_pdf(path, source)
    raise RagError(f"Формат не поддерживается: {path.name}. Нужен .md, .txt или .pdf")


def _load_pdf(path: Path, source: str) -> list[RawDocument]:
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise RagError(f"Не удалось прочитать PDF {path.name}: {exc}") from exc
    if reader.is_encrypted:
        raise RagError(f"PDF защищён паролем и пропущен: {path.name}")

    documents: list[RawDocument] = []
    for page_number, page in enumerate(reader.pages, start=1):
        extracted = page.extract_text() or ""
        documents.append(
            RawDocument(
                text=extracted,
                source=source,
                page=page_number,
                kind="pdf",
            )
        )
    return documents


def collect_files(path: Path) -> tuple[list[Path], Path]:
    """Вернуть список файлов и корень, от которого считаются имена в ссылках."""
    if not path.exists():
        raise RagError(f"Путь не найден: {path}")
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise RagError(f"Формат не поддерживается: {path.name}")
        return [path], path.parent
    if path.is_dir():
        files = [
            item
            for item in sorted(path.rglob("*"))
            if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES
        ]
        if not files:
            raise RagError(f"В папке нет .md, .txt или .pdf: {path}")
        return files, path
    raise RagError(f"Путь не является файлом или папкой: {path}")


def load_documents(paths: list[Path]) -> list[RawDocument]:
    documents: list[RawDocument] = []
    found_any = False
    for raw_path in paths:
        files, root = collect_files(raw_path)
        found_any = True
        for file_path in files:
            documents.extend(load_file(file_path, root))
    if not found_any:
        raise RagError("Не переданы пути к документам")
    if not any(document.text.strip() for document in documents):
        raise RagError("В документах нет текста. Сканы PDF без текстового слоя не читаются.")
    return documents
