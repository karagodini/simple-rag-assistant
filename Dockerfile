FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag ./rag
COPY examples ./examples
COPY README.md LICENSE ./

ENV PYTHONUNBUFFERED=1 \
    RAG_INDEX_DIR=/data/rag_index

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
