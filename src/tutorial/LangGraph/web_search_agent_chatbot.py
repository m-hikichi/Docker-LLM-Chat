import json
import os
from typing import Annotated, List, Literal, Tuple

import gradio as gr
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel
from typing_extensions import TypedDict

from llms.llm_api_inference import fetch_model_from_llm_api


class State(TypedDict):
    messages: Annotated[list, add_messages]
    temperature: float
    max_tokens: int


@tool
def web_search(query: str):
    """WEB検索を行い、関連する情報を取得するツール"""
    wrapper = DuckDuckGoSearchAPIWrapper(region="jp-jp", max_results=3)
    web_search = DuckDuckGoSearchResults(
        api_wrapper=wrapper,
        source="text",
        output_format="list",
    )

    return web_search.invoke(query)


def select_next_agent(state: State) -> Literal["chat_agent", "web_search_agent"]:
    class routeResponse(BaseModel):
        """
        ユーザの入力に対して回答を生成する際に知識が必要な場合は`web_serch_agent`、知識が必要ない単純な会話のみの場合は`chat_agent`を呼び出してください。
        """

        next: Literal["chat_agent", "web_search_agent"]

    # LLMモデルを初期化し、パラメータを設定
    llm = fetch_model_from_llm_api(
        llm_type="ollama",
        model=os.environ["LLM_API_MODEL_NAME"],
        temperature=0,
    )
    structured_output_llm = llm.with_structured_output(routeResponse)

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                "あなたは次のノードを決定するAIエージェントです。\n"
                "ユーザーの入力から適切な次のノードを選択してください。\n"
                "次のノードには以下があります：\n"
                "- chat_agent\n"
                "- web_search_agent"
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    chain = prompt | structured_output_llm
    result = chain.invoke(state)
    return result.next


def web_search_agent(state: State):
    # LLMモデルを初期化し、パラメータを設定
    llm = fetch_model_from_llm_api(
        llm_type="ollama",
        model=os.environ["LLM_API_MODEL_NAME"],
        temperature=state["temperature"],
        max_tokens=state["max_tokens"],
    )
    llm_with_tools = llm.bind_tools([web_search])

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                "あなたはWEB検索を行うAIエージェントです。"
                "ユーザーの入力に対して適切な回答を返してください。"
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    chain = prompt | llm_with_tools
    response = chain.invoke(state)
    return {"messages": response}


def chat_agent(state: State):
    # LLMモデルを初期化し、パラメータを設定
    llm = fetch_model_from_llm_api(
        llm_type="ollama",
        model=os.environ["LLM_API_MODEL_NAME"],
        temperature=state["temperature"],
        max_tokens=state["max_tokens"],
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                "あなたは誠実で優秀なAIエージェントです。"
                "ユーザーの入力に対して楽しく会話を続けてください。"
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    chain = prompt | llm
    response = chain.invoke(state)
    return {"messages": response}


async def generate_chat_response(
    message: str,
    history: List[Tuple[str, str]],
    max_tokens: int,
    temperature: float,
):
    # 状態グラフを作成
    graph = StateGraph(State)
    # ノードの作成
    graph.add_node("chat_agent", chat_agent)
    graph.add_node("web_search_agent", web_search_agent)
    graph.add_node("web_search", ToolNode([web_search]))
    # エッジの作成
    graph.add_conditional_edges(
        START, select_next_agent, ["chat_agent", "web_search_agent"]
    )
    graph.add_edge("chat_agent", END)
    graph.add_conditional_edges(
        "web_search_agent", tools_condition, {"tools": "web_search", "__end__": END}
    )
    graph.add_edge("web_search", "web_search_agent")

    runner = graph.compile()

    # プロンプトを構築
    messages = []
    if history:
        for user, assistant in history:
            messages.append(HumanMessage(user))
            messages.append(AIMessage(assistant))
    messages.append(HumanMessage(message))

    response_text = ""
    reference = ""
    async for event in runner.astream_events(
        {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        version="v2",
    ):
        if event["event"] == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                response_text += content
                yield response_text + reference
        elif event["event"] == "on_tool_start":
            yield "WEB検索中..."
        elif event["event"] == "on_tool_end":
            if event["name"] == "web_search":
                for search_result in json.loads(event["data"]["output"].content):
                    reference += (
                        f"\n[{search_result['title']}]({search_result['link']})"
                    )
            yield "回答生成中..."


def build_chat_ui():
    chatbot = gr.Chatbot(avatar_images=["icons/user.png", "icons/elyza.png"])

    chat_interface = gr.ChatInterface(
        fn=generate_chat_response,
        chatbot=chatbot,
        additional_inputs=[
            gr.Slider(
                minimum=1, maximum=2048, value=512, step=1, label="最大出力トークン数"
            ),
            gr.Slider(
                minimum=0.0, maximum=1.0, value=0.2, step=0.1, label="Temperature"
            ),
        ],
        additional_inputs_accordion=gr.Accordion(label="詳細設定", open=False),
        title="WEB検索チャットボット",
        submit_btn="送信",
    )

    return chat_interface


if __name__ == "__main__":
    build_chat_ui().launch(server_name="0.0.0.0")
