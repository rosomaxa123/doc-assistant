import os
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from dotenv import load_dotenv

# Загружаем .env
load_dotenv()

# 1. Загрузка локальной модели эмбеддингов
print("Загрузка модели эмбеддингов...")
model = SentenceTransformer("./models/rubert-tiny2")


# 2. Чтение датасета
def load_dataset():
    print("Загрузка датасета...")
    try:
        df = pd.read_csv("dataset_clean.csv", dtype=str, on_bad_lines='skip')
        texts = df['text'].dropna().tolist()
        # Фильтруем пустые строки
        chunks = [t.strip() for t in texts if len(t.strip()) > 50]
        print(f"Успешно загружено {len(chunks)} чанков (книг).")
        return chunks
    except Exception as e:
        print(f"ОШИБКА ЗАГРУЗКИ: {e}")
        return ["Ошибка чтения файла."]


all_chunks = load_dataset()


# 3. Построение индекса
def build_index():
    print("Создание эмбеддингов и индекса...")
    chunk_embeddings = model.encode(all_chunks, show_progress_bar=True)
    chunk_embeddings = np.array(chunk_embeddings).astype('float32')
    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(chunk_embeddings)
    return index


# 4. ГИБРИДНЫЙ ПОИСК (По смыслу + По ключевым словам)
def get_context(question, index, k=1):
    # А) Поиск по ключевым словам (чтобы найти точные совпадения)
    # Берем слова длиной больше 3 символов
    words = [w.lower().strip('?,.!') for w in question.split() if len(w) > 3]

    keyword_matches = []
    for chunk in all_chunks:
        if any(word in chunk.lower() for word in words):
            keyword_matches.append(chunk)
            if len(keyword_matches) >= k:
                break

    if keyword_matches:
        print("Найдено точное совпадение по словам!")
        return keyword_matches

    # Б) Векторный поиск (если точных слов нет, ищем по смыслу)
    print("Поиск по смыслу через FAISS...")
    q_embedding = model.encode([question])
    q_embedding = np.array(q_embedding).astype('float32')
    distances, indices = index.search(q_embedding, k)
    return [all_chunks[i] for i in indices[0]]


# 5. Генерация ответа
def rag_answer(question, index, k=1):
    # Берем только 1 лучший кусок, чтобы не пугать фильтры GigaChat
    context_chunks = get_context(question, index, k=k)
    context_text = context_chunks[0]

    # Безопасный промпт (просим пересказать, а не отвечать)
    prompt = f"""Прочитай следующий текст из базы данных и кратко перескажи его суть:

Текст:
{context_text}

Пересказ:"""

    auth_key = os.getenv("LLM_API_KEY")
    if not auth_key:
        raise ValueError("Ключ LLM_API_KEY не найден в .env")

    giga = GigaChat(
        credentials=auth_key,
        scope="GIGACHAT_API_PERS",
        model="GigaChat-2",
        base_url="https://api.giga.chat/v1",
        verify_ssl_certs=False
    )

    try:
        response = giga.chat(
            Chat(messages=[Messages(role=MessagesRole.USER, content=prompt)])
        )
        answer = response.choices[0].message.content
    except Exception as e:
        # Если GigaChat все равно капризничает, выдаем текст напрямую
        answer = f"Найденный текст в базе:\n\n{context_text}"

    return answer, context_chunks