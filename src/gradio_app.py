import os
from typing import List, Tuple

import gradio as gr
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from llms.llm_api import fetch_llm_api_model


def generate_chat_response(
    message: str,
    history: List[Tuple[str, str]],
    system_prompt: str,
    max_tokens: int,
    temperature: float,
):
    # load llm model
    llm_model = fetch_llm_api_model(
        model=os.environ["LLM_API_MODEL_NAME"],
    )

    # update parameters
    llm_model.max_tokens = max_tokens
    llm_model.temperature = temperature

    # construct prompt
    messages = []
    messages.append(SystemMessage(system_prompt))
    if history:
        for user, assistant in history:
            messages.append(HumanMessage(user))
            messages.append(AIMessage(assistant))
    messages.append(HumanMessage(message))

    prompt = ChatPromptTemplate.from_messages(messages)

    # construct chain
    chain = prompt | llm_model

    # inference
    response = ""
    for text in chain.stream({}):
        response += text.content
        yield response


def build_chat_ui():
    chatbot = gr.Chatbot(
        show_copy_button=True,
        avatar_images=["icons/user.png", "icons/elyza.png"],
    )

    accordion = gr.Accordion(
        label="詳細設定",
        open=False,
    )

    system_prompt_textbox = gr.Textbox(
        value="あなたは誠実で優秀な日本人のアシスタントです。",
        label="システムプロンプト",
    )

    max_tokens_slider = gr.Slider(
        minimum=1,
        maximum=2048,
        value=512,
        step=1,
        label="最大出力トークン数",
    )

    temperature_slider = gr.Slider(
        minimum=0.1,
        maximum=1.0,
        value=0.2,
        step=0.1,
        label="Temperature",
    )

    chat_interface = gr.ChatInterface(
        fn=generate_chat_response,
        chatbot=chatbot,
        additional_inputs=[
            system_prompt_textbox,
            max_tokens_slider,
            temperature_slider,
        ],
        additional_inputs_accordion=accordion,
        title="Chat with LLM",
        submit_btn="↑",
        # retry_btn="🔄 同じ入力でもう一度生成",
        # undo_btn="↩️ ひとつ前の状態に戻る",
        # clear_btn="🗑️ これまでの出力を消す",
    )

    return chat_interface


if __name__ == "__main__":
    demo = build_chat_ui()
    demo.launch(server_name="0.0.0.0")
