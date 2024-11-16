import os
from typing import Literal, Optional, Union

import requests
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


class LLMAPIException(Exception):
    def __init__(self, api_url: str, response_status_code: Optional[int] = None):
        if response_status_code is None:
            self.message = f"LLM API {api_url} is not available."
        else:
            self.message = f"LLM API {api_url} is not available. Status code: {response_status_code}"

    def __str__(self):
        return self.message


def fetch_model_from_llm_api(
    llm_type: Literal["openai", "ollama"] = "openai",
    model: str = "gpt-3.5-turbo-instruct",
    temperature: float = 0.7,
    max_tokens: int = 256,
) -> Union[ChatOpenAI, ChatOllama]:
    """
    Args:
        llm_type : The type of language model to use, either "openai" or "ollama".
        model : Model name to use.
        temperature : What sampling temperature to use.
        max_tokens : The maximum number of tokens to generate in the completion. -1 returns as many tokens as possible given the prompt and the models maximal context size.
    """
    try:
        # Making an API request to fetch models
        response = requests.get(os.environ["OPENAI_API_BASE"] + "/v1/models")
        response.raise_for_status()  # Automatically handles non-200 status codes

        if llm_type == "openai":
            # Assuming ChatOpenAI is a class that needs these parameters
            llm_model = ChatOpenAI(
                base_url=os.environ["OPENAI_API_BASE"] + "/v1",
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        elif llm_type == "ollama":
            # Assuming ChatOllama is a class that needs these parameters
            llm_model = ChatOllama(
                base_url=os.environ["OPENAI_API_BASE"],
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            # Raise error for unsupported llm_type
            raise ValueError(
                f"Unsupported llm_type '{llm_type}'. Supported types are 'openai' and 'ollama'."
            )

        return llm_model

    except requests.exceptions.RequestException as e:
        raise LLMAPIException(
            os.environ["OPENAI_API_BASE"],
            e.response.status_code if e.response else None,
        )


if __name__ == "__main__":
    # load llm model
    llm_model = fetch_model_from_llm_api(
        model=os.environ["LLM_API_MODEL_NAME"],
    )

    # create prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage("あなたは誠実で優秀な日本人のアシスタントです。"),
            HumanMessagePromptTemplate.from_template("{query}"),
        ]
    )

    # construct chain
    chain = prompt | llm_model

    # inference
    for text in chain.stream({"query": "こんにちは"}):
        print(text.content, flush=True, end="")
    print()
