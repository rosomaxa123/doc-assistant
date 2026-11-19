import sys
import os
import datetime
import json
from src.rag import build_index, get_context

print("Загрузка индекса для tool_search...")
INDEX = build_index()

def tool_search(query: str) -> str:
    """Поиск релевантных фрагментов в базе документов.
    Используй, когда нужен факт, определение или цитата из корпуса.
    Не используй для арифметики."""
    # Ищем только 1 лучший чанк
    chunks = get_context(query, INDEX, k=1)
    if not chunks:
        return "В документах не найдено."
    return "\n".join(chunks)

def tool_calc(expression: str) -> str:
    """Вычислить арифметическое выражение, например '2*(3+4)'.
    Используй только для чисел и операторов + - * / ( )."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Ошибка вычисления: {e}"

def tool_time() -> str:
    """Возвращает текущую дату и время.
    Используй, когда спрашивают про 'сегодня', 'сейчас' или текущую дату."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Ровно 3 инструмента
TOOLS = {
    "search": tool_search,
    "calc": tool_calc,
    "time": tool_time
}

def run_tool(name: str, args: dict) -> str:
    if name not in TOOLS:
        return f"Ошибка: неизвестный инструмент '{name}'."
    try:
        if isinstance(args, str):
            if name == "calc":
                return TOOLS[name](args)
            try:
                args = json.loads(args)
            except:
                return TOOLS[name](args)
        return TOOLS[name](**args)
    except Exception as e:
        return f"Ошибка инструмента {name}: {e}"

# Схемы для Native-подхода (GigaChat)
TOOL_SCHEMAS = [
    {
        "name": "search",
        "description": "Поиск в базе документов. Используй для вопросов о книгах.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    },
    {
        "name": "calc",
        "description": "Калькулятор для математики.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}
    },
    {
        "name": "time",
        "description": "Текущее время.",
        "parameters": {"type": "object", "properties": {}}
    }
]