import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
import os

# Загружаем модель эмбеддингов один раз при старте
model = SentenceTransformer("./models/rubert-tiny2")

# Тестовые данные (можешь заменить на чтение файла)
text_data = """
Искусственный интеллект (ИИ) — это область информатики, которая занимается созданием интеллектуальных машин. 
RAG (Retrieval-Augmented Generation) — это технология, которая позволяет языковым моделям отвечать на вопросы, используя внешние базы знаний.
Векторная база данных хранит информацию в виде чисел (векторов), что позволяет искать по смыслу, а не по точному совпадению слов.
"""

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

all_chunks = chunk_text(text_data)

def build_index():
    """Строит индекс FAISS"""
    print("Создание эмбеддингов и индекса...")
    chunk_embeddings = model.encode(all_chunks)
    chunk_embeddings = np.array(chunk_embeddings).astype('float32')
    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(chunk_embeddings)
    return index

def get_context(question, index, k=1):
    q_embedding = model.encode([question])
    q_embedding = np.array(q_embedding).astype('float32')
    distances, indices = index.search(q_embedding, k)
    return [all_chunks[i] for i in indices[0]]

def rag_answer(question, index, k=1):
    """Возвращает ответ LLM и список источников"""
    context_chunks = get_context(question, index, k)
    context_text = "\n\n".join(context_chunks)
    
    prompt = f"""Ответь на вопрос ТОЛЬКО по приведённым фрагментам.
Если ответа в них нет — скажи 'в документах не найдено'.

Фрагменты:
{context_text}

Вопрос: {question}"""

    # Ключ берем из переменной окружения (как требует Докер)
    auth_key = os.getenv("LLM_API_KEY")
    
    giga = GigaChat(
        credentials=auth_key, 
        scope="GIGACHAT_API_PERS",
        model="GigaChat-2", 
        base_url="https://api.giga.chat/v1",
        verify_ssl_certs=False
    )
    
    response = giga.chat(Chat(messages=[Messages(role=MessagesRole.USER, content=prompt)]))
    answer = response.choices[0].message.content
    
    return answer, context_chunks