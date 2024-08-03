import requests
from typing import Optional
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import Field, SecretStr


class LLMAPIException(Exception):
    def __init__(self, api_url: str, response_status_code: Optional[int] = None):
        if response_status_code is None:
            self.message = f"LLM API {api_url} is not available."
        else:
            self.message = f"LLM API {api_url} is not available. Status code: {response_status_code}"

    def __str__(self):
        return self.message


def fetch_llm_api_model(
    api_url: Optional[str] = Field(default=None, alias="base_url"),
    api_key: Optional[SecretStr] = Field(default=None, alias="api_key"),
    model: str = Field(default="gpt-3.5-turbo-instruct", alias="model"),
    temperature: float = 0.7,
    batch_size: int = 20,
    max_tokens: int = 256,
    top_p: float = 1.0,
) -> OpenAI:
    """
    Args:
        api_url : Base URL path for API requests, leave blank if not using a proxy or service emulator.
        api_key : Automatically inferred from env var `OPENAI_API_KEY` if not provided.
        model : Model name to use.
        temperature : What sampling temperature to use.
        batch_size : Batch size to use when passing multiple documents to generate.
        max_tokens : The maximum number of tokens to generate in the completion. -1 returns as many tokens as possible given the prompt and the models maximal context size.
        top_p : Total probability mass of tokens to consider at each step.
    """
    try:
        response = requests.get(api_url + "/models")
        if response.status_code == 200:
            llm_model = OpenAI(
                openai_api_base=api_url,
                openai_api_key=api_key,
                model=model,
                temperature=temperature,
                batch_size=batch_size,
                max_tokens=max_tokens,
                top_p=top_p,
            )

            return llm_model
        else:
            raise LLMAPIException(api_url, response.status_code)

    except requests.exceptions.RequestException:
        raise LLMAPIException(api_url)


if __name__ == "__main__":
    # load llm model
    llm_model = fetch_llm_api_model(
        api_url="http://ollama:11434/v1",
        api_key="dummy_api_key",
        model="ELYZA:8B-Q4_K_M",
    )

    # create prompt
    prompt = PromptTemplate(
        input_variables=["query"],
        template="""<|start_header_id|>system<|end_header_id|>

あなたは誠実で優秀な日本人のアシスタントです。<|eot_id|><|start_header_id|>user<|end_header_id|>

{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

""",
    )

    # construct chain
    chain = prompt | llm_model

    # inference
    for text in chain.stream({"query": "有名な犬種をリスト形式で教えてください"}):
        print(text, flush=True, end="")
