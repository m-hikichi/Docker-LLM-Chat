# reference: https://langfuse.com/docs/prompts/get-started

import os

from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langfuse import Langfuse

from llms.llm_api_inference import fetch_model_from_llm_api

os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-************************************"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-************************************"
os.environ["LANGFUSE_HOST"] = "http://langfuse-server:3000"


if __name__=="__main__":
    # initialize Langfuse client
    langfuse = Langfuse()

    # get prompt from langfuse
    # optional parameters: https://langfuse.com/docs/prompts/get-started#:~:text=from_messages(langfuse_prompt.get_langchain_prompt())-,Optional%20parameters,-%23%20Get%20specific%20version
    langfuse_prompt = langfuse.get_prompt("sample", type="chat", version=1)
    langchain_prompt = ChatPromptTemplate.from_messages(langfuse_prompt.get_langchain_prompt())

    # load llm model
    llm_model = fetch_model_from_llm_api(
        model=os.environ["LLM_API_MODEL_NAME"],
    )

    # construct chain
    chain = langchain_prompt | llm_model | StrOutputParser()

    for text in chain.stream({"story": "トムとジェリー"}):
        print(text, flush=True, end="")
    print()
