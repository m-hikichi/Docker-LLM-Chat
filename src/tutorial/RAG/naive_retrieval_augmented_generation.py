import asyncio
import os
from typing import Annotated, List

import langchain_core
from langchain.schema import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from embeddings.huggingface_embedding import load_embedding_model
from llms.llm_api_inference import fetch_model_from_llm_api
from retrieve.retrieve import (
    construct_hybrid_retriever,
    load_documents,
    split_documents,
)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def retrieve_documents(state: State):
    def format_docs(docs: List[langchain_core.documents.Document]) -> str:
        return f"\n{'-'*100}\n".join(
            [f"Document {i+1}:\n\n" + doc.page_content for i, doc in enumerate(docs)]
        )

    embedding_model = load_embedding_model(
        model_path="/workspace/models/multilingual-e5-large",
    )

    documents = load_documents("/workspace/documents")

    splitted_documents = split_documents(
        documents=documents,
        separators=["。"],
        chunk_size=256,
        chunk_overlap=0,
    )

    hybrid_retriever = construct_hybrid_retriever(
        documents=splitted_documents,
        embedding_model=embedding_model,
        semantic_k=2,
        keyword_k=2,
    )

    retrieve_chain = hybrid_retriever | format_docs

    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            query = message.content

    return {
        "messages": ToolMessage(
            content=retrieve_chain.invoke(query),
            tool_call_id="",
        )
    }


def llm_agent(state: State):
    llm_model = fetch_model_from_llm_api(
        model=os.environ["LLM_API_MODEL_NAME"],
        temperature=0.2,
        max_tokens=2048,
    )

    messages = ChatPromptTemplate.from_messages(state["messages"])

    chain = messages | llm_model | StrOutputParser()

    return {"messages": chain.invoke({})}


if __name__ == "__main__":
    #
    graph = StateGraph(State)

    # add node
    graph.add_node("retrieve_documents", retrieve_documents)
    graph.add_node("llm_agent", llm_agent)

    # add edge
    graph.add_edge(START, "retrieve_documents")
    graph.add_edge("retrieve_documents", "llm_agent")
    graph.add_edge("llm_agent", END)

    # compile graph
    runner = graph.compile()

    messages = [
        SystemMessage(
            "あなたは誠実で優秀な日本人のアシスタントです。以下の「コンテキスト情報」を元に「質問」に回答してください。\n"
            "なおコンテキスト情報に無い情報は回答に含めないでください。\n"
            "コンテキスト情報から回答が導けない場合は「わかりません」と回答してください。"
        ),
        HumanMessage("イーブイからニンフィアに進化させる方法を教えてください。"),
    ]

    async def run_chat():
        async for event in runner.astream_events({"messages": messages}, version="v1"):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                print(content, flush=True, end="")

    asyncio.run(run_chat())
    print()
