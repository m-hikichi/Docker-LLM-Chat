from llama_cpp import Llama


llm = Llama(
    model_path="/models/ELYZA-japanese-Llama-2-13b-fast-instruct-gguf/ELYZA-japanese-Llama-2-13b-fast-instruct-q5_K_M.gguf",
    n_gpu_layers=-1,
)


def construct_prompt(system_prompt, message, history):
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


def generate_response_from_chat(prompt):
    output = llm.create_completion(
        prompt=prompt,
        max_tokens=4096,
        temperature=0.2,
        top_p=0.95,
        top_k=40,
        min_p=0.05,
        typical_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        repeat_penalty=1.1,
        seed=None,
    )
    return output["choices"][0]["text"]
