from llama_cpp import Llama


llm = Llama(model_path="/models/ELYZA-japanese-Llama-2-13b-fast-instruct-gguf/ELYZA-japanese-Llama-2-13b-fast-instruct-q5_K_M.gguf", n_gpu_layers=-1)


def create_prompt(system_prompt, message, history):
    prompt = []
    prompt.append({"role": "system", "content": system_prompt})

    if history:
        for user_context, assistant_context in history:
            prompt.append({"role": "user", "content": user_context})
            prompt.append({"role": "assistant", "content": assistant_context})
    prompt.append({"role": "user", "content": message})

    return prompt


def chat_llama2(prompt):
    # Reference: https://github.com/abetlen/llama-cpp-python
    output = llm.create_chat_completion(messages=prompt)
    return output["choices"][0]["message"]["content"]
