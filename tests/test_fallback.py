from pathlib import Path

from rag.answer import compose_answer
from rag.config import NOT_FOUND, Settings
from rag.models import Chunk, Hit
from rag.pipeline import RagPipeline


def _pipeline(tmp_path: Path, **overrides) -> RagPipeline:
    examples = Path("examples/docs")
    fields = {
        "index_dir": tmp_path / "index",
        "openai_api_key": None,
        "openai_base_url": None,
        "min_score": 0.25,
    }
    fields.update(overrides)
    settings = Settings(**fields)
    pipeline = RagPipeline(settings)
    pipeline.ingest([examples])
    return pipeline


def test_without_api_key_answer_is_extractive_and_cited(tmp_path: Path):
    pipeline = _pipeline(tmp_path)
    answer = pipeline.ask(
        "Сколько календарных дней составляет ежегодный оплачиваемый отпуск "
        "и за сколько дней подаётся заявление?"
    )
    assert answer.mode == "extractive"
    assert answer.found
    assert "28" in answer.answer
    assert "14" in answer.answer
    assert "reglament.md" in answer.answer
    assert "фрагмент" in answer.answer
    assert answer.sources
    assert answer.sources[0].source == "reglament.md"


def test_price_question_cites_price_file(tmp_path: Path):
    pipeline = _pipeline(tmp_path)
    answer = pipeline.ask("Сколько стоит чат-бот с ответами по документам?")
    assert answer.found
    assert "120 000" in answer.answer or "120000" in answer.answer
    assert any(source.source == "price.txt" for source in answer.sources)
    assert "фрагмент" in answer.answer


def test_unknown_question_says_not_found(tmp_path: Path):
    pipeline = _pipeline(tmp_path)
    for question in (
        "Какой сегодня курс биткоина на бирже?",
        "Как зовут генерального директора компании?",
    ):
        answer = pipeline.ask(question)
        assert answer.found is False
        assert answer.answer == NOT_FOUND
        assert answer.sources == []
        assert answer.mode == "extractive"


def test_low_score_hits_do_not_become_an_answer():
    settings = Settings(openai_api_key=None, min_score=0.5)
    weak = Hit(
        chunk=Chunk(id=1, text="Совсем другой текст про кофе.", source="faq.md"),
        score=0.11,
    )
    answer = compose_answer("курс биткоина", [weak], settings)
    assert answer.found is False
    assert answer.answer == NOT_FOUND


def test_llm_error_falls_back_to_extractive(monkeypatch, tmp_path: Path):
    pipeline = _pipeline(tmp_path, min_score=0.05)
    pipeline.settings.openai_api_key = "test-key"
    pipeline.settings.openai_base_url = "http://127.0.0.1:9/v1"

    def explode(*_args, **_kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr("rag.answer.generate_with_llm", explode)
    answer = pipeline.ask("Какие суточные в командировке по России?")
    assert answer.mode == "extractive"
    assert answer.found
    assert "1500" in answer.answer
    assert "фрагмент" in answer.answer
    assert "Внешняя модель недоступна" in answer.answer
