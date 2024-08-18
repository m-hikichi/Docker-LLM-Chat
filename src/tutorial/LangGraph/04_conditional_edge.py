from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage, HumanMessage


llm = ChatOpenAI(
    base_url="http://ollama:11434/v1",
    api_key="dummy-api-key",
    model="ELYZA:8B-Q4_K_M",
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
    state["messages"].append(ToolMessage(content=tool_output, tool_call_id=last_message.tool_calls[0]["id"]))
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
