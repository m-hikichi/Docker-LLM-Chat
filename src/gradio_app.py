import gradio as gr
from chat_elyza import construct_prompt, generate_response_from_chat


def chat(message, history):
    prompt = construct_prompt("あなたは誠実で優秀な日本人のアシスタントです。", message, history)
    return generate_response_from_chat(prompt)


def build_chat_ui():
    with gr.Blocks() as demo:
        markdown = gr.Markdown(
            """
            # Chat with ELYZA
            """
        )
        chat_interface = gr.ChatInterface(chat)

    return demo


if __name__ == "__main__":
    demo = build_chat_ui()
    demo.launch(server_name="0.0.0.0")
