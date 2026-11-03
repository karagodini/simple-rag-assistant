from pathlib import Path

from fastapi.testclient import TestClient

from rag.api import create_app
from rag.config import NOT_FOUND, Settings


def test_health_and_chat_page(tmp_path: Path):
    client = TestClient(create_app(Settings(index_dir=tmp_path / "index", openai_api_key=None)))
    assert client.get("/health").json() == {"status": "ok"}
    page = client.get("/")
    assert page.status_code == 200
    assert "Ассистент по документам" in page.text


def test_ask_before_ingest_is_rejected(tmp_path: Path):
    client = TestClient(create_app(Settings(index_dir=tmp_path / "missing", openai_api_key=None)))
    response = client.post("/ask", json={"question": "сколько дней отпуска?"})
    assert response.status_code == 400
    assert "Индекс не найден" in response.json()["detail"]


def test_ingest_and_ask_over_http(tmp_path: Path):
    client = TestClient(create_app(Settings(index_dir=tmp_path / "index", openai_api_key=None)))
    ingested = client.post("/ingest", json={"path": "examples/docs"})
    assert ingested.status_code == 200
    assert ingested.json()["chunks"] > 0

    found = client.post(
        "/ask",
        json={"question": "Какая скидка постоянному клиенту?"},
    )
    assert found.status_code == 200
    body = found.json()
    assert body["found"] is True
    assert body["mode"] == "extractive"
    assert "10" in body["answer"]
    assert any(source["source"] == "faq.md" for source in body["sources"])

    missing = client.post("/ask", json={"question": "Как зовут генерального директора?"})
    assert missing.status_code == 200
    assert missing.json()["answer"] == NOT_FOUND
    assert missing.json()["sources"] == []


def test_upload_txt(tmp_path: Path):
    client = TestClient(create_app(Settings(index_dir=tmp_path / "index", openai_api_key=None)))
    payload = "Суточные по России составляют 1500 рублей.".encode()
    response = client.post(
        "/ingest/upload",
        files={"files": ("note.txt", payload, "text/plain")},
    )
    assert response.status_code == 200
    answer = client.post("/ask", json={"question": "Какие суточные по России?"})
    assert answer.status_code == 200
    assert answer.json()["found"] is True
    assert "1500" in answer.json()["answer"]
