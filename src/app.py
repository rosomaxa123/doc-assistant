from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from src.rag import build_index, rag_answer

app = FastAPI(title="Doc Assistant")

# Глобальная переменная для индекса
index_state = None

# Модель данных для валидации входящего запроса
class Query(BaseModel):
    question: str
    k: int = 3

@app.on_event("startup")
def load():
    """Загружаем индекс один раз при старте сервера"""
    global index_state
    print("Запуск: построение индекса FAISS...")
    index_state = build_index()
    print("Индекс готов!")

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(q: Query):
    if not q.question.strip():
        raise HTTPException(status_code=400, detail="Пустой вопрос")

    try:
        # Вызов RAG
        answer, sources = rag_answer(q.question, index_state, k=q.k)
        return {"answer": answer, "sources": sources}
    except Exception as e:
        # Возвращаем текст ошибки прямо в ответе API
        raise HTTPException(status_code=500, detail=str(e))