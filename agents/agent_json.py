import os, json, re, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from agents.tools import run_tool

auth_key = os.getenv("LLM_API_KEY")
giga = GigaChat(credentials=auth_key, scope="GIGACHAT_API_PERS", model="GigaChat-2", verify_ssl_certs=False)

SYSTEM = (
    "Ты агент с инструментами: search(query), calc(expression), time().\n"
    "На каждом шаге верни СТРОГО ОДИН JSON-объект:\n"
    '{"action":"search|calc|time|final","input":"...","answer":"..."}\n'
    "Для action=final заполни answer, для остальных — input.\n"
    "НЕ пиши несколько JSON подряд. НЕ добавляй пояснений."
)


def extract_json(text: str) -> dict:
    start_idx = text.find('{')
    if start_idx == -1: raise ValueError("Нет JSON")
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(text[start_idx:])
        return obj
    except json.JSONDecodeError as e:
        raise ValueError(f"Ошибка JSON: {e}")


def agent_json(question, max_steps=5):
    history = f"Вопрос: {question}\n"
    for step_num in range(max_steps):
        print(f"\n[step {step_num}] Запрос...")
        raw = giga.chat(Chat(messages=[Messages(role=MessagesRole.USER, content=SYSTEM + "\n" + history)])).choices[
            0].message.content
        print(f"[step {step_num}] Ответ: {raw[:100]}...")

        try:
            step = extract_json(raw)
        except ValueError as e:
            history += f"Ошибка: {e}. Верни чистый JSON.\n"
            continue

        action = str(step.get("action", "")).lower()
        if action == "final":
            # Берем текст либо из answer, либо из input
            final_ans = step.get("answer") or step.get("input") or "Нет ответа"
            return final_ans

        input_val = step.get("input", "")
        print(f"[step {step_num}] Действие: {action}({input_val})")
        result = run_tool(action, input_val)
        print(f"[step {step_num}] Результат: {str(result)[:100]}...")
        history += f"Действие: {action}({input_val}) -> {result}\n"

    return "Не удалось получить ответ."


if __name__ == "__main__":
    question = "О чем книга, где героиня нашла скарб и купила будку?"
    print(f"--- Запрос: {question} ---")
    answer = agent_json(question)
    print(f"\n--- ФИНАЛЬНЫЙ ОТВЕТ ---\n{answer}")