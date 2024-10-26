# reference: https://github.com/langchain-ai/langgraph/blob/ed0f55af85542f7debbc86821f0dc1b547e7ee51/examples/streaming-tokens.ipynb

import asyncio
from typing import Annotated, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict


llm_model = ChatOllama(
    base_url="http://ollama:11434",
    api_key="dummy-api-key",
    model="ELYZA:8B-Q4_K_M",
    temperature=0,
)


@tool
def search(query: str):
    """パーソナル情報を格納したデータベースを検索するAPI"""
    return ["にゃんたは釣りを行っているYouTuberです"]


tools = [search]
llm_model = llm_model.bind_tools(tools)


class State(TypedDict):
    messages: Annotated[list, add_messages]


# モデルを呼び出す関数の定義
async def llm_agent(state: State):
    # 注: Python < 3.11 では、RunnableConfig をノードで受け取り、これを llm.ainvoke(..., config) の 2 番目の引数として明示的に設定を渡す必要がある
    messages = state["messages"]
    response = await llm_model.ainvoke(messages)
    # 既存のリストに追加されるので、リストを返す
    return {"messages": response}


# ツールを単純なToolNodeでラップ
# これは、tool_callsを持つAIMessagesを含むメッセージのリストを取り込み、ツールを実行し、出力をToolMessagesとして返す単純なクラス
tool_node = ToolNode(tools)


# 続行するかどうかを決定する関数の定義
def should_continue(state: State) -> Literal["__end__", "tools"]:
    messages = state["messages"]
    last_message = messages[-1]
    # 関数呼び出しがなければ、終了
    if not last_message.tool_calls:
        return END
    # そうでない場合は、続ける
    else:
        return "tools"


# Graph の作成
graph = StateGraph(State)

# 定義した Node の追加
graph.add_node("agent", llm_agent)
graph.add_node("tools", tool_node)

# entry_point の定義
graph.add_edge(START, "agent")
graph.add_conditional_edges(
    # 開始ノードを定義。ここでは `agent` を使用。
    # これは `agent` ノードが呼び出された後のエッジを意味する。
    "agent",
    # 次に呼び出されるノードを決定する関数を渡す
    should_continue,
)
graph.add_edge("tools", "agent")

# Graph のコンパイル
runner = graph.compile()


async def main():
    messages = [
        SystemMessage("あなたは誠実で優秀な日本人のアシスタントです。"),
        HumanMessage("にゃんたについて教えて"),
    ]

    async for event in runner.astream_events(
        {"messages": messages},
        version="v1",
    ):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            print(content, flush=True, end="")
        elif kind == "on_tool_start":
            print("-" * 10)
            print(
                f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}"
            )
        elif kind == "on_tool_end":
            print(f"Done tool: {event['name']}")
            print(f"Tool output was: {event['data'].get('output')}")
            print("-" * 10)


if __name__ == "__main__":
    asyncio.run(main())
    print()
