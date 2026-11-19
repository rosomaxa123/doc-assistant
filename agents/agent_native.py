import os, json, sys
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.tools import TOOL_SCHEMAS, run_tool

auth_key = os.getenv("LLM_API_KEY")
giga = GigaChat(credentials=auth_key, scope="GIGACHAT_API_PERS", model="GigaChat-2", verify_ssl_certs=False)


def run_agent(user_query: str, max_iterations: int = 5):
    print(f"\n--- Запрос: {user_query} ---")
    messages = [Messages(role=MessagesRole.USER, content=user_query)]

    for step in range(max_iterations):
        print(f"[step {step}] Отправка запроса в LLM...")
        response = giga.chat(Chat(messages=messages, functions=TOOL_SCHEMAS))
        msg = response.choices[0].message

        if not msg.function_call:
            print(f"[step {step}] Финальный ответ.")
            return msg.content

        call = msg.function_call
        print(f"[step {step}] Вызов: {call.name}({call.arguments})")
        if isinstance(call.arguments, str):
            args = json.loads(call.arguments) if call.arguments else {}
        else:
            args = call.arguments
        result = run_tool(call.name, args)
        print(f"[step {step}] Результат: {str(result)[:100]}...")

        messages.append(msg)
        import json
        json_result = json.dumps({"result": str(result)}, ensure_ascii=False)
        messages.append(Messages(role=MessagesRole.FUNCTION, content=json_result, name=call.name))

    return "Не удалось получить ответ."


if __name__ == "__main__":
    # Вопрос строго по датасету
    question = "О чем книга, где героиня нашла скарб и купила будку?"
    answer = run_agent(question)
    print(f"\nФинальный ответ:\n{answer}")