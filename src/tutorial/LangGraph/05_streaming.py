import asyncio
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig


llm_model = ChatOpenAI(
    base_url="http://ollama:11434/v1",
    api_key="dummy-api-key",
    model="ELYZA:8B-Q4_K_M",
    temperature=0,
)


class State(TypedDict):
    messages: Annotated[list, add_messages]


async def chatbot(state: State, config: RunnableConfig):
    messages = state["messages"]
    response = await llm_model.ainvoke(messages, config)

    return {"messages": response}


# Graph の作成
graph = StateGraph(State)

# 定義した Node の追加
graph.add_node("chatbot", chatbot)

# entry_point と finish_point の定義
graph.set_entry_point("chatbot")
graph.set_finish_point("chatbot")

# Graph のコンパイル
runner = graph.compile()


async def main():
    messages = [
        SystemMessage("あなたは誠実で優秀な日本人のアシスタントです。"),
        HumanMessage("魔法少女まどか☆マギカの登場キャラクターについて教えて"),
    ]

    async for event in runner.astream_events(
        {"messages": messages},
        version="v1",
    ):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            print(content, flush=True, end="")


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
print()
