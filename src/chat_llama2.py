from llama_cpp import Llama


llm = Llama(model_path="/tmp/llama2-model/llama-2-7b-chat.ggmlv3.q5_K_M.bin")

# Reference: https://github.com/abetlen/llama-cpp-python
print(
    llm.create_chat_completion(
        messages = [
            {"role": "system", "content": "You are an assistant."},
            {"role": "user", "content": "My name is Taro Yamada. My birthday is January 1. Nice to meet you."},
            {'role': 'assistant', 'content': 'Nice to meet you too, Taro! How can I assist you today? Is there anything you would like to know or any task you would like me to help you with?'},
            {"role": "user", "content": "Do you remember my birthday?"},
        ]
    )
)
