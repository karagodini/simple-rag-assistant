"""Командная строка: ingest, ask, serve."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.config import Settings, load_settings
from rag.errors import RagError
from rag.pipeline import RagPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag",
        description="Ассистент по вашим документам: ответы со ссылками на фрагменты.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Прочитать документы и собрать индекс")
    ingest.add_argument("paths", nargs="+", help="Файлы или папки с .md, .txt, .pdf")
    _add_common(ingest)
    ingest.add_argument(
        "--embedder",
        choices=["tfidf", "sbert"],
        default=None,
        help="tfidf работает без сети, sbert — мультиязычные эмбеддинги",
    )

    ask = sub.add_parser("ask", help="Задать вопрос по уже построенному индексу")
    ask.add_argument("question", help="Вопрос в кавычках")
    _add_common(ask)
    ask.add_argument("--top-k", type=int, default=None, help="Сколько фрагментов искать")

    serve = sub.add_parser("serve", help="Запустить HTTP API и страницу чата")
    _add_common(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    return parser


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--index-dir",
        default=None,
        help="Папка индекса (по умолчанию .rag_index или RAG_INDEX_DIR)",
    )


def _apply_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    if args.index_dir:
        settings.index_dir = Path(args.index_dir)
    embedder = getattr(args, "embedder", None)
    if embedder:
        settings.embedder = embedder
    return settings


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = _apply_overrides(load_settings(), args)
    pipeline = RagPipeline(settings)

    try:
        if args.command == "ingest":
            report = pipeline.ingest([Path(path) for path in args.paths])
            names = ", ".join(report["file_names"])
            print(
                f"Готово. Файлов: {report['files']} ({names}). "
                f"Фрагментов: {report['chunks']}. "
                f"Эмбеддер: {report['embedder']}. Индекс: {report['index_dir']}"
            )
            return 0
        if args.command == "ask":
            result = pipeline.ask(args.question, top_k=args.top_k)
            print(result.answer)
            if result.sources:
                print("\nИсточники:")
                for source in result.sources:
                    print(f"- {source.citation()} (схожесть {source.score:.2f})")
            print(f"\nРежим: {result.mode}")
            return 0
        if args.command == "serve":
            import uvicorn

            from rag.api import create_app

            app = create_app(settings)
            uvicorn.run(app, host=args.host, port=args.port)
            return 0
    except RagError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130

    print("Неизвестная команда", file=sys.stderr)
    return 2
