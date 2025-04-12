import asyncio
import os
from typing import Annotated

import langchain_core
from langchain.schema import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from typing_extensions import TypedDict

from embeddings.embedding_api import fetch_embedding_model
from llms.llm_api_inference import fetch_model_from_llm_api
from retrieve.retrieve import (
    construct_hybrid_retriever,
    load_documents,
    split_documents,
)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    search_query: str
    retriever: langchain_core.retrievers.BaseRetriever


def generate_search_query(state: State):
    class SearchKeywords(BaseModel):
        keywords: list[str]

    llm = fetch_model_from_llm_api(
        model=os.environ["LLM_API_MODEL_NAME"],
        temperature=0.2,
        max_tokens=2048,
    )
    structured_output_llm = llm.with_structured_output(SearchKeywords)

    preprocessed_search_query = state["search_query"]
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                "あなたは入力された文章から、検索キーワードを抽出する作業を行っています。\n"
                "入力されたメッセージから、キーワードをリスト形式のテキストで出力してください。"
            ),
            HumanMessage(preprocessed_search_query),
        ]
    )

    chain = prompt | structured_output_llm
    result = chain.invoke({})

    if result is None:
        return {"search_query": preprocessed_search_query}
    else:
        return {"search_query": " ".join(result.keywords)}


def retrieve_documents(state: State):
    messages = state["messages"]
    search_query = state["search_query"]
    retriever = state["retriever"]

    retrieve_documents = retriever.invoke(search_query)
    for doc in retrieve_documents:
        messages.append(
            ToolMessage(
                content=doc.page_content,
                tool_call_id="",
            )
        )

    return {"messages": messages}


def llm_agent(state: State):
    llm = fetch_model_from_llm_api(
        model=os.environ["LLM_API_MODEL_NAME"],
        temperature=0.2,
        max_tokens=2048,
    )

    messages = [
        SystemMessage(
            "あなたは誠実で優秀な日本人のアシスタントです。以下の「コンテキスト情報」を元に「質問」に回答してください。\n"
            "なおコンテキスト情報に無い情報は回答に含めないでください。\n"
            "コンテキスト情報から回答が導けない場合は「わかりません」と回答してください。"
        )
    ]
    messages.extend(state["messages"])
    prompt = ChatPromptTemplate.from_messages(messages)

    chain = prompt | llm | StrOutputParser()

    return {"messages": chain.invoke({})}


if __name__ == "__main__":
    embedding_model = fetch_embedding_model(
        base_url="http://ollama:11434",
        model_name="multilingual-e5-large",
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

    # init graph
    graph = StateGraph(State)

    # add node
    graph.add_node("generate_search_query", generate_search_query)
    graph.add_node("retrieve_documents", retrieve_documents)
    graph.add_node("llm_agent", llm_agent)

    # add edge
    graph.add_edge(START, "generate_search_query")
    graph.add_edge("generate_search_query", "retrieve_documents")
    graph.add_edge("retrieve_documents", "llm_agent")
    graph.add_edge("llm_agent", END)

    # compile graph
    runner = graph.compile()

    question = "イーブイからニンフィアに進化させる方法を教えてください。"
    messages = [
        HumanMessage(question),
    ]

    async def run_chat():
        async for event in runner.astream_events(
            {
                "messages": messages,
                "search_query": question,
                "retriever": hybrid_retriever,
            },
            version="v2",
        ):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                print(content, flush=True, end="")

    asyncio.run(run_chat())
    print()
