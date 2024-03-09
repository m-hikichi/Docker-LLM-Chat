import logging
from typing import List, Optional, Union, Iterator
from llama_cpp import Llama
from llama_cpp.llama_types import (
    CreateCompletionResponse,
    CreateCompletionStreamResponse,
)


sh = logging.StreamHandler()
sh_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s:%(message)s (%(name)s)", datefmt="%Y/%m/%d %p %I:%M:%S"
)
sh.setLevel(logging.DEBUG)
sh.setFormatter(sh_formatter)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(sh)


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

    logger.debug(prompt)
    return prompt


def generate_response_from_chat(
    prompt: Union[str, List[int]],
    max_tokens: Optional[int] = 16,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 40,
    min_p: float = 0.05,
    typical_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    repeat_penalty: float = 1.1,
    stream: bool = False,
    seed: Optional[int] = None,
) -> Union[
    Iterator[CreateCompletionResponse],
    Iterator[CreateCompletionStreamResponse],
]:
    return llm.create_completion(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        typical_p=typical_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        repeat_penalty=repeat_penalty,
        stream=stream,
        seed=seed,
    )
