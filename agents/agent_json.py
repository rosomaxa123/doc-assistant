import os
import json
import re
import sys

# Добавляем корень проекта в пути поиска
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from agents.tools import run_tool

# Инициализация GigaChat
auth_key = os.getenv("LLM_API_KEY",
                     "MDFhMWVmNTUtNTJmZi03MGZmLWJmOTctNjVjZjFjOTZiMmEyOmEzOTI5MmQ3LTQzOWUtNDgwMC04ZjcyLTM3MWI2ODBlYzJiYg==")
giga = GigaChat(credentials=auth_key, scope="GIGACHAT_API_PERS", model="GigaChat-2", verify_ssl_certs=False)

# Ужесточенный системный промпт
SYSTEM = (
    "Ты агент с инструментами: search(query), calc(expression), time().\n"
    "На каждом шаге верни СТРОГО ОДИН JSON-объект и ничего больше:\n"
    '{"action":"search|calc|time|final","input":"...","answer":"..."}\n'
    "Для action=final заполни answer, для остальных — input (для time input оставь пустым).\n"
    "НЕ пиши несколько JSON подряд. НЕ добавляй пояснений. Только один JSON.\n"
    "Используй инструменты, пока не соберёшь ответ."
)


# Надежный парсер, который берет только первый JSON-объект
def extract_json(text: str) -> dict:
    start_idx = text.find('{')
    if start_idx == -1:
        raise ValueError("В ответе модели нет JSON")

    decoder = json.JSONDecoder()
    try:
        # raw_decode возвращает (объект, индекс_конца), игнорируя всё после первого JSON
        obj, end_idx = decoder.raw_decode(text[start_idx:])
        return obj
    except json.JSONDecodeError as e:
        raise ValueError(f"Ошибка парсинга JSON: {e}")


# Основной цикл JSON-агента
def agent_json(question, max_steps=5):
    history = f"Вопрос: {question}\n"

    for step_num in range(max_steps):
        print(f"\n[step {step_num}] Отправка запроса в LLM...")

        try:
            raw_response = giga.chat(Chat(messages=[Messages(role=MessagesRole.USER, content=SYSTEM + "\n" + history)]))
            raw = raw_response.choices[0].message.content
        except Exception as e:
            print(f"[step {step_num}] Ошибка запроса к LLM: {e}")
            return "Ошибка LLM"

        print(f"[step {step_num}] Сырой ответ модели: {raw[:150]}...")

        try:
            step = extract_json(raw)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"[step {step_num}] Ошибка парсинга JSON: {e}")
            history += f"Ошибка: ответ не парсится как JSON ({e}). Верни ровно один чистый JSON.\n"
            continue

        action = str(step.get("action", "")).strip().lower()

        if action == "final":
            print(f"[step {step_num}] Финальный ответ получен.")
            return step.get("answer", "Модель не дала ответ")

        input_value = step.get("input", "")
        print(f"[step {step_num}] Модель выбрала действие: {action}({input_value})")

        # Выполняем инструмент
        result = run_tool(action, input_value)
        print(f"[step {step_num}] Результат инструмента: {str(result)[:100]}...")

        # Добавляем в историю
        history += f"Действие: {action}({input_value}) -> {result}\n"

    return "Не удалось получить ответ за отведенное число шагов."


if __name__ == "__main__":
    question = "Что такое RAG и посчитай 2+2"
    print(f"--- Запрос: {question} ---")
    answer = agent_json(question)
    print(f"\n--- ФИНАЛЬНЫЙ ОТВЕТ ---\n{answer}")