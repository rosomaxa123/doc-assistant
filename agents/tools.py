import sys
import os
import datetime
import json
from src.rag import build_index, get_context

# Загружаем индекс один раз
print("Загрузка индекса для tool_search...")
INDEX = build_index()


def tool_search(query: str) -> str:
    """Поиск релевантных фрагментов в базе документов.
    Используй, когда нужен факт, определение или цитата из корпуса.
    Не используй для арифметики."""
    chunks = get_context(query, INDEX, k=2)
    return "\n".join(chunks)


def tool_calc(expression: str) -> str:
    """Вычислить арифметическое выражение, например '2*(3+4)'.
    Используй только для чисел и операторов + - * / ( )."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Ошибка вычисления: {e}"


def tool_time() -> str:
    """Возвращает текущую дату и время.
    Используй, когда спрашивают про 'сегодня', 'сейчас' или текущую дату."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TOOLS = {
    "search": tool_search,
    "calc": tool_calc,
    "time": tool_time
}


def run_tool(name: str, args: dict) -> str:
    if name not in TOOLS:
        return f"Ошибка: неизвестный инструмент '{name}'. Доступны: {list(TOOLS.keys())}."
    try:
        # Если args пришел как строка (бывает в JSON-подходе)
        if isinstance(args, str):
            # Если это calc, возвращаем строку, для остальных пытаемся распарсить
            if name == "calc":
                return TOOLS[name](args)
            try:
                args = json.loads(args)
            except:
                return TOOLS[name](args)  # Передаем как есть

        # Если это словарь, распаковываем
        return TOOLS[name](**args)
    except Exception as e:
        return f"Ошибка инструмента {name}: {e}"


# Схемы для GigaChat (плоский формат, без вложенного "type": "function")
TOOL_SCHEMAS = [
    {
        "name": "search",
        "description": "Поиск релевантных фрагментов в базе документов. Используй для фактов и определений.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Поисковый запрос"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "calc",
        "description": "Вычислить арифметическое выражение. Используй для математики.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Выражение, например '2+2'"}
            },
            "required": ["expression"]
        }
    },
    {
        "name": "time",
        "description": "Возвращает текущую дату и время.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
]