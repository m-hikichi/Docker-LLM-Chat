import os
from typing import Annotated, List, Literal, Tuple

import gradio as gr
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from llms.llm_api import fetch_llm_api_model

class State(TypedDict):
    messages: Annotated[list, add_messages]


@tool
def search(query: str):
    """WEB検索を行い、関連する情報を取得するツール"""
    wrapper = DuckDuckGoSearchAPIWrapper(region="jp-jp", max_results=3)
    search = DuckDuckGoSearchResults(api_wrapper=wrapper, source="text")

    return search.invoke(query)


async def generate_chat_response(
    message: str,
    history: List[Tuple[str, str]],
    system_prompt: str,
    max_tokens: int,
    temperature: float,
):
    # LLMモデルを初期化し、パラメータを設定
    llm_model = fetch_llm_api_model(
        llm_type="ollama",
        model=os.environ["LLM_API_MODEL_NAME"],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # ツールとノードを設定
    tools = [search]
    llm_model = llm_model.bind_tools(tools)
    tool_node = ToolNode(tools)

    # エージェント関数の定義
    async def llm_agent(state: State):
        messages = state["messages"]
        response = await llm_model.ainvoke(messages)
        return {"messages": response}

    # 継続条件を決定する関数の定義
    def should_continue(state: State) -> Literal["__end__", "tools"]:
        last_message = state["messages"][-1]
        return "tools" if last_message.tool_calls else END

    # 状態グラフを作成
    graph = StateGraph(State)
    graph.add_node("agent", llm_agent)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, ["tools", END])
    graph.add_edge("tools", "agent")

    runner = graph.compile()

    # プロンプトを構築
    messages = []
    messages.append(SystemMessage(system_prompt))
    if history:
        for user, assistant in history:
            messages.append(HumanMessage(user))
            messages.append(AIMessage(assistant))
    messages.append(HumanMessage(message))

    response_text = ""
    async for event in runner.astream_events({"messages": messages}, version="v1"):
        if event["event"] == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                response_text += content
                yield response_text
        elif event["event"] == "on_tool_start":
            yield "WEB検索中..."
        elif event["event"] == "on_tool_end":
            yield "回答生成中..."


def build_chat_ui():
    chatbot = gr.Chatbot(avatar_images=["icons/user.png", "icons/elyza.png"])

    chat_interface = gr.ChatInterface(
        fn=generate_chat_response,
        chatbot=chatbot,
        additional_inputs=[
            gr.Textbox(
                value="あなたは誠実で優秀な日本人のアシスタントです。",
                label="システムプロンプト",
            ),
            gr.Slider(
                minimum=1, maximum=2048, value=512, step=1, label="最大出力トークン数"
            ),
            gr.Slider(
                minimum=0.1, maximum=1.0, value=0.2, step=0.1, label="Temperature"
            ),
        ],
        additional_inputs_accordion=gr.Accordion(label="詳細設定", open=False),
        title="Llama-3-ELYZA-JP-8B-demo",
        submit_btn="送信",
    )

    return chat_interface


if __name__ == "__main__":
    build_chat_ui().launch(server_name="0.0.0.0")
