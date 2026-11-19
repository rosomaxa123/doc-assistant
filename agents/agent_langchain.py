import os, sys
from typing import Any, List, Optional

# Добавляем корень проекта в пути поиска
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from agents.tools import tool_search, tool_calc, tool_time
from langchain_core.tools import Tool

# Обертка для GigaChat
class GigaChatLLM(LLM):
    giga: Any
    @property
    def _llm_type(self) -> str: return "gigachat"
    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
        response = self.giga.chat(Chat(messages=[Messages(role=MessagesRole.USER, content=prompt)]))
        return response.choices[0].message.content

auth_key = os.getenv("LLM_API_KEY")
giga_client = GigaChat(credentials=auth_key, scope="GIGACHAT_API_PERS", model="GigaChat-2", verify_ssl_certs=False)

# Инструменты (строго 3 штуки, как в задании)
tools = [
    Tool(name="search", func=tool_search, description="Поиск в базе книг. Используй для поиска сюжетов и персонажей."),
    Tool(name="calc", func=tool_calc, description="Калькулятор. Используй только для математики."),
    Tool(name="time", func=tool_time, description="Текущее время и дата.")
]

# Строгий промпт без лишних слов
template = """Ты — ИИ-агент. Отвечай на вопросы, используя только доступные инструменты: {tools}

Отвечай СТРОГО в следующем формате:
Question: вопрос пользователя
Thought: подумай, какой инструмент использовать
Action: одно из [{tool_names}]
Action Input: входные данные для инструмента
Observation: результат работы инструмента
... (повторяй Thought/Action/Action Input/Observation, если нужно)
Thought: Я знаю финальный ответ
Final Answer: финальный ответ на вопрос

ВАЖНО: 
1. Как только ты нашел информацию, сразу пиши "Final Answer: ..." и НЕ вызывай инструменты снова.
2. НЕ пиши лишних пояснений, приветствий или заголовков. Только строго по формату.

Question: {input}
{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)
llm = GigaChatLLM(giga=giga_client)
agent = create_react_agent(llm, tools, prompt)

# Ограничиваем до 4 шагов, чтобы не было бесконечного цикла
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=4
)

if __name__ == "__main__":
    question = "О чем книга, где героиня нашла скарб и купила будку?"
    print(f"--- Запрос: {question} ---")
    try:
        result = agent_executor.invoke({"input": question})
        print(f"\n--- ФИНАЛЬНЫЙ ОТВЕТ ---\n{result['output']}")
    except Exception as e:
        print(f"\nОшибка выполнения агента: {e}")