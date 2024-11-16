import asyncio
import os
from typing import Annotated

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from llms.llm_api_inference import fetch_model_from_llm_api

llm_model = fetch_model_from_llm_api(
    llm_type="ollama",
    model=os.environ["LLM_API_MODEL_NAME"],
    temperature=0,
)


class State(TypedDict):
    messages: Annotated[list, add_messages]


async def chatbot(state: State):
    messages = state["messages"]
    response = await llm_model.ainvoke(messages)

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


if __name__ == "__main__":
    asyncio.run(main())
    print()
