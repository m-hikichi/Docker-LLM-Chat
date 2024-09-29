# reference: https://github.com/langchain-ai/langgraph/blob/main/docs/docs/how-tos/streaming-tokens.ipynb

import asyncio
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessageChunk, SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool


llm_model = ChatOpenAI(
    base_url="http://ollama:11434/v1",
    api_key="dummy-api-key",
    model="ELYZA:8B-Q4_K_M",
    temperature=0,
)


@tool
def search(query: str):
    """パーソナル情報を格納したデータベースを検索するAPI"""
    return ["にゃんたは最新技術について紹介しているYouTuberです"]

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

    first = True
    async for msg, metadata in runner.astream({"messages": messages}, stream_mode="messages"):
        if msg.content and not isinstance(msg,HumanMessage):
            print(msg.content, flush=True, end="")

        if isinstance(msg, AIMessageChunk):
            if first:
                gathered = msg
                first = False
            else:
                gathered = gathered + msg

            if msg.tool_call_chunks:
                print(gathered.tool_calls)


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
print()
