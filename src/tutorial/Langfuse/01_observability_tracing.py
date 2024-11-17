import os

from langchain.schema import StrOutputParser
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langfuse.callback import CallbackHandler
from llms.llm_api_inference import fetch_model_from_llm_api

os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-************************************"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-************************************"
os.environ["LANGFUSE_HOST"] = "http://langfuse-server:3000"


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
    chain = prompt | llm_model | StrOutputParser()

    #
    langfuse_handler = CallbackHandler()
    # tests the SDK connection with the server
    langfuse_handler.auth_check()

    # inference
    for text in chain.stream(
        {"query": "こんにちは"}, config={"callbacks": [langfuse_handler]}
    ):
        print(text, flush=True, end="")
    print()
