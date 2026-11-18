import os
import json
import sys
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

# Добавляем корень проекта в пути поиска, чтобы импорты работали
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.tools import TOOL_SCHEMAS, run_tool

# Инициализация GigaChat
auth_key = os.getenv("LLM_API_KEY",
                     "MDFhMWVmNTUtNTJmZi03MGZmLWJmOTctNjVjZjFjOTZiMmEyOmEzOTI5MmQ3LTQzOWUtNDgwMC04ZjcyLTM3MWI2ODBlYzJiYg==")
giga = GigaChat(credentials=auth_key, scope="GIGACHAT_API_PERS", model="GigaChat-2", verify_ssl_certs=False)


def run_agent(user_query: str, max_iterations: int = 5):
    print(f"\n--- Запрос: {user_query} ---")
    messages = [Messages(role=MessagesRole.USER, content=user_query)]

    for step in range(max_iterations):
        print(f"[step {step}] Отправка запроса в LLM...")

        try:
            # Используем функции (tools) в GigaChat
            response = giga.chat(Chat(messages=messages, functions=TOOL_SCHEMAS))
        except Exception as e:
            print(f"Ошибка запроса к GigaChat: {e}")
            return "Ошибка LLM"

        msg = response.choices[0].message

        # Проверяем, просит ли модель инструмент
        if not msg.function_call:
            print(f"[step {step}] Финальный ответ получен.")
            return msg.content

        # Модель просит вызвать инструмент
        call = msg.function_call
        print(f"[step {step}] Модель просит вызвать: {call.name} с аргументами {call.arguments}")

        # Выполняем инструмент
        try:
            args = json.loads(call.arguments) if call.arguments else {}
        except:
            args = call.arguments  # Если пришла строка

        result = run_tool(call.name, args)
        print(f"[step {step}] Результат инструмента: {str(result)[:100]}...")

        # Добавляем в историю ответ модели и результат инструмента
        messages.append(msg)
        messages.append(Messages(role=MessagesRole.FUNCTION, content=str(result), name=call.name))

    return "Не удалось получить ответ за отведенное число шагов."


if __name__ == "__main__":
    question = "Что такое RAG и посчитай 2*(3+4)"
    answer = run_agent(question)
    print(f"\nФинальный ответ: {answer}")