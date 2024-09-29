from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI


llm_model = ChatOpenAI(
    base_url="http://ollama:11434/v1",
    api_key="dummy-api-key",
    model="ELYZA:8B-Q4_K_M",
    temperature=0,
)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    api_call_count: int

# Node の作成
def chatbot(state: State):
    return {
        "messages": [llm_model.invoke(state["messages"])],
        "api_call_count": state["api_call_count"] + 1,
    }

# Graph の作成
graph = StateGraph(State)

# 定義した Node の追加
graph.add_node("chatbot", chatbot)

# entry_point と finish_point の定義
graph.set_entry_point("chatbot")
graph.set_finish_point("chatbot")

# Graph のコンパイル
runner = graph.compile()
response = runner.invoke({"messages": ["こんにちは"], "api_call_count": 0})
print(response)
print(response["messages"][-1].content)
