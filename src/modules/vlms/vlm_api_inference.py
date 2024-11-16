from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from llms.llm_api_inference import fetch_model_from_llm_api


if __name__ == "__main__":
    # load vlm
    vlm = fetch_model_from_llm_api(
        llm_type="ollama",
        model="llama3.2-vision",
    )

    # create prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage("あなたは誠実で優秀な日本人のアシスタントです。"),
            HumanMessage(
                content=[
                    {"type": "text", "text": "画像内に犬が何匹映っていますか"},
                    {"type": "image_url", "image_url": {"url": "dogs.jpg"}}
                ]
            ),
        ]
    )

    # construct chain
    chain = prompt | vlm

    # inference
    for text in chain.stream({}):
        print(text.content, flush=True, end="")
    print()

