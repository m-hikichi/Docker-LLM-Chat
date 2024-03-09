import gradio as gr
import elyza


def elyza_chat_wrapper(
    message, history, system_prompt, max_tokens, temperature, top_p, top_k
):
    prompt = elyza.construct_prompt(system_prompt, message, history)
    result = elyza.generate_response_from_chat(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    return result["choices"][0]["text"]


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
        fn=elyza_chat_wrapper,
        chatbot=chatbot,
        additional_inputs=[
            system_prompt_textbox,
            max_tokens_slider,
            temperature_slider,
            top_p_slider,
            top_k_slider,
        ],
        additional_inputs_accordion=accordion,
        title="ELYZA-japanese-Llama-2-13b-fast-instruct-gguf-demo",
        submit_btn="送信",
        retry_btn="🔄 同じ入力でもう一度生成",
        undo_btn="↩️ ひとつ前の状態に戻る",
        clear_btn="🗑️ これまでの出力を消す",
    )

    return chat_interface


if __name__ == "__main__":
    demo = build_chat_ui()
    demo.launch(server_name="0.0.0.0")
