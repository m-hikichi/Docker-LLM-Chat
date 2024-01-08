import gradio as gr
from chat_llama2 import create_prompt, chat_llama2


def chat_response(message, history):
    prompt = create_prompt("You are an assistant.", message, history)
    return chat_llama2(prompt)


def build_chat_ui():
    with gr.Blocks() as demo:
        markdown = gr.Markdown(
            """
            # Chat with Llama2
            """
        )
        chatinterface = gr.ChatInterface(chat_response)

    return demo


if __name__ == "__main__":
    demo = build_chat_ui()
    demo.launch(server_name="0.0.0.0")
