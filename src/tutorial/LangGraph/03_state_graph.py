import os
from typing import Annotated

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from llms.llm_api_inference import fetch_model_from_llm_api

llm_model = fetch_model_from_llm_api(
    model=os.environ["LLM_API_MODEL_NAME"],
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
