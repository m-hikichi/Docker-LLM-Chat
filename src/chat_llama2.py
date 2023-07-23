from llama_cpp import Llama


model = Llama(model_path="/tmp/llama2-model/llama-2-7b-chat.ggmlv3.q8_0.bin")

while True:
    input_text = input("> ")
    input_text = "user: " + input_text + "com:"

    output = model(
        input_text,
        max_tokens=1024,
        stop=["user:", "com:"],
        echo=True,
    )

    input_text += " "
    print("> " + output["choices"][0]["text"].replace(input_text, "") + "\n")
