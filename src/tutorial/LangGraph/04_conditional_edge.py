import os
from typing import Annotated

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from llms.llm_api import fetch_llm_api_model

llm = fetch_llm_api_model(
    model=os.environ["LLM_API_MODEL_NAME"],
    temperature=0,
)


@tool
def fake_database_api(query: str) -> str:
    """パーソナル情報を格納したデータベースを検索するAPI"""
    # return "にゃんたは最新技術について紹介しているYouTuberです"
    return "にゃんたは釣りを行っているYouTuberです"


class State(TypedDict):
    messages: Annotated[list, add_messages]


llm_with_tools = llm.bind_tools([fake_database_api])


def llm_agent(state):
    state["messages"].append(llm_with_tools.invoke(state["messages"]))
    return state


def tool(state):
    tool_by_name = {"fake_database_api": fake_database_api}
    last_message = state["messages"][-1]
    tool_function = tool_by_name[last_message.tool_calls[0]["name"]]
    tool_output = tool_function.invoke(last_message.tool_calls[0]["args"])
    state["messages"].append(
        ToolMessage(content=tool_output, tool_call_id=last_message.tool_calls[0]["id"])
    )
    return state


def router(state):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool"
    else:
        return "__end__"


# Graph の作成
graph = StateGraph(State)

# 定義した Node の追加
graph.add_node("llm_agent", llm_agent)
graph.add_node("tool", tool)

graph.set_entry_point("llm_agent")
graph.add_conditional_edges("llm_agent",
                            router,
                            {"tool":"tool", "__end__": END})

graph.add_edge("tool", "llm_agent")

runner = graph.compile()

print(
    runner.invoke(
        {"messages": [HumanMessage(content="にゃんたについて教えて")]},
        debug=True,
    )
)
