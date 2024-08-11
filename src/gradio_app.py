import gradio as gr
from typing import List, Tuple
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llms.llm_api import fetch_llm_api_model


llm_model = fetch_llm_api_model(
    api_url="http://ollama:11434/v1",
    api_key="dummy_api_key",
    model="ELYZA:8B-Q4_K_M",
)


def construct_llama2_prompt(
    system_prompt: str,
    message: str,
    history: List[Tuple[str, str]],
):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    B_OS, E_OS = "<s>", "</s>"

    prompt = "{bos_token}{b_inst} {system}".format(
        bos_token=B_OS, b_inst=B_INST, system=f"{B_SYS}{system_prompt}{E_SYS}"
    )

    if history:
        for user, assistant in history:
            prompt += (
                "{user} {e_inst}\n{assistant} {eos_token}\n{bos_token}{b_inst}".format(
                    user=user,
                    e_inst=E_INST,
                    assistant=assistant,
                    eos_token=E_OS,
                    bos_token=B_OS,
                    b_inst=B_INST,
                )
            )

    prompt += "{message} {e_inst}".format(
        message=message,
        e_inst=E_INST,
    )

    return prompt


def construct_llama3_prompt(
    system_prompt: str,
    message: str,
    history: List[Tuple[str, str]],
):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    B_OS, E_OS = "<s>", "</s>"

    prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"

    if history:
        for user, assistant in history:
            prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
            prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>"

    prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    return prompt


def generate_chat_response(
    message: str,
    history: List[Tuple[str, str]],
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
):
    # update parameters
    llm_model.max_tokens = max_tokens
    llm_model.temperature = temperature
    llm_model.top_p = top_p
    # llm_model.top_k = top_k

    # construct prompt
    prompt = PromptTemplate(
        input_variables=[],
        template=construct_llama3_prompt(system_prompt, message, history),
    )

    # construct chain
    chain = prompt | llm_model | StrOutputParser()

    # inference
    response = ""
    for text in chain.stream({}):
        response += text
        yield response


def build_chat_ui():
    chatbot = gr.Chatbot(avatar_images=["icons/user.png", "icons/elyza.png"])

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

    top_p_slider = gr.Slider(
        minimum=0.05,
        maximum=1.00,
        value=0.95,
        step=0.05,
        label="Top-p",
    )

    top_k_slider = gr.Slider(
        minimum=1,
        maximum=1000,
        value=50,
        label="Top-k",
    )

    chat_interface = gr.ChatInterface(
        fn=generate_chat_response,
        chatbot=chatbot,
        additional_inputs=[
            system_prompt_textbox,
            max_tokens_slider,
            temperature_slider,
            top_p_slider,
            top_k_slider,
        ],
        additional_inputs_accordion=accordion,
        title="Llama-3-ELYZA-JP-8B-demo",
        submit_btn="送信",
        retry_btn="🔄 同じ入力でもう一度生成",
        undo_btn="↩️ ひとつ前の状態に戻る",
        clear_btn="🗑️ これまでの出力を消す",
    )

    return chat_interface


if __name__ == "__main__":
    demo = build_chat_ui()
    demo.launch(server_name="0.0.0.0")
