import os
import sys
from typing import Any, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from agents.tools import tool_search, tool_calc, tool_time
from langchain_core.tools import Tool

# 1. Создаем обертку для GigaChat, чтобы LangChain понимал, как с ним общаться
class GigaChatLLM(LLM):
    giga: Any

    @property
    def _llm_type(self) -> str:
        return "gigachat"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        response = self.giga.chat(Chat(messages=[Messages(role=MessagesRole.USER, content=prompt)]))
        return response.choices[0].message.content

# Инициализация
auth_key = os.getenv("LLM_API_KEY", "MDFhMWVmNTUtNTJmZi03MGZmLWJmOTctNjVjZjFjOTZiMmEyOmEzOTI5MmQ3LTQzOWUtNDgwMC04ZjcyLTM3MWI2ODBlYzJiYg==") 
giga_client = GigaChat(credentials=auth_key, scope="GIGACHAT_API_PERS", model="GigaChat-2", verify_ssl_certs=False)

# 2. Оборачиваем наши функции в инструменты LangChain
tools = [
    Tool(name="search", func=tool_search, description="Поиск в базе документов. Используй для фактов."),
    Tool(name="calc", func=tool_calc, description="Калькулятор. Используй для математики."),
    Tool(name="time", func=tool_time, description="Текущее время и дата.")
]

# 3. Промпт для агента (ReAct формат)
template = """Отвечай на вопросы, используя инструменты. У тебя есть доступ к: {tools}

Используй следующий формат:
Question: вопрос, на который нужно ответить
Thought: подумай, что делать
Action: одно из [{tool_names}]
Action Input: входные данные для действия
Observation: результат действия
... (повторяй Thought/Action/Action Input/Observation, если нужно)
Thought: Теперь я знаю финальный ответ
Final Answer: финальный ответ на вопрос

Question: {input}
{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)

# 4. Создаем агента
llm = GigaChatLLM(giga=giga_client)
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

if __name__ == "__main__":
    question = "Что такое RAG и посчитай 2+2"
    print(f"--- Запрос: {question} ---")
    # Запуск агента
    result = agent_executor.invoke({"input": question})
    print(f"\n--- ФИНАЛЬНЫЙ ОТВЕТ ---\n{result['output']}")