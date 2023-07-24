from llama_cpp import Llama


model = Llama(model_path="/tmp/llama2-model/llama-2-7b-chat.ggmlv3.q8_0.bin")
conversation_history = ""

while True:
    user_input = input("> ")
    conversation_history += "user: " + user_input + "\nllama:"

    output = model(
        conversation_history,
        max_tokens=1024,
        stop=["user:", "llama:"],
        echo=True,
    )

    conversation_history += " "
    llama_reply = output["choices"][0]["text"].replace(conversation_history, "")
    print("> " + llama_reply + "\n")
    conversation_history = output["choices"][0]["text"]
