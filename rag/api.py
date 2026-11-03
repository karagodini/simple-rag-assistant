"""HTTP API: POST /ingest, POST /ask и простая страница чата."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rag.config import Settings, load_settings
from rag.errors import RagError
from rag.pipeline import RagPipeline

STATIC_DIR = Path(__file__).resolve().parent / "static"
ALLOWED_UPLOADS = {".md", ".txt", ".pdf"}


class IngestRequest(BaseModel):
    path: str = Field(description="Файл или папка на машине, где запущен сервер")


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or load_settings()
    app = FastAPI(
        title="Ассистент по документам",
        description="Мини-RAG: ответы по вашим файлам со ссылками на фрагменты.",
        version="0.1.0",
    )
    app.state.settings = resolved
    app.state.pipeline = RagPipeline(resolved)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/")
    def chat_page() -> FileResponse:
        page = STATIC_DIR / "index.html"
        if not page.exists():
            raise HTTPException(status_code=404, detail="Страница чата не найдена")
        return FileResponse(page)

    @app.post("/ingest")
    def ingest(body: IngestRequest) -> dict:
        try:
            return app.state.pipeline.ingest([Path(body.path)])
        except RagError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/ingest/upload")
    def ingest_upload(files: list[UploadFile] = File(...)) -> dict:
        if not files:
            raise HTTPException(status_code=400, detail="Не переданы файлы")
        temporary = Path(tempfile.mkdtemp(prefix="rag-upload-"))
        try:
            saved = 0
            for upload in files:
                name = Path(upload.filename or "document.txt").name
                suffix = Path(name).suffix.lower()
                if suffix not in ALLOWED_UPLOADS:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Формат не поддерживается: {name}",
                    )
                target = temporary / name
                target.write_bytes(upload.file.read())
                saved += 1
            if saved == 0:
                raise HTTPException(status_code=400, detail="Не переданы файлы")
            return app.state.pipeline.ingest([temporary])
        except RagError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    @app.post("/ask")
    def ask(body: AskRequest) -> dict:
        try:
            answer = app.state.pipeline.ask(body.question, top_k=body.top_k)
        except RagError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return answer.to_dict()

    return app


app = create_app()
