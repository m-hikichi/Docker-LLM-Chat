import asyncio
import os
from typing import Annotated

import langchain_core
from bs4 import BeautifulSoup
from googlesearch import search as google_search
from langchain.schema import StrOutputParser
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from embeddings.embedding_api import fetch_embedding_model
from llms.llm_api_inference import fetch_model_from_llm_api
from retrieve.retrieve import (
    construct_hybrid_retriever,
    construct_semantic_retriever,
    load_documents,
    split_documents,
)
from naive_retrieval_augmented_generation import (
    generate_search_query,
    retrieve_documents,
)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    search_query: str
    retriever: langchain_core.retrievers.BaseRetriever


def check_search_online_requirement(state: State):
    class ToolPerform(BaseModel):
        search_online: bool = Field(description="whether to perform a web search")

    llm_model = fetch_model_from_llm_api(
        model=os.environ["LLM_API_MODEL_NAME"],
        temperature=0,
        max_tokens=2048,
    )
    structured_output_llm = llm_model.with_structured_output(ToolPerform)

    messages = [
        SystemMessage(
            "質問に対して参照情報から正しい回答を導けるかどうかを判断してください\n"
            "もし導けない場合は、`search_online`ツールを用いてWEBから情報を取得してください"
        )
    ]
    messages.extend(state["messages"])
    prompt = ChatPromptTemplate.from_messages(messages)

    chain = prompt | structured_output_llm
    response = chain.invoke({})

    if response is None:
        return "search_online"
    if response.search_online:
        return "search_online"
    else:
        return "llm_agent"


def search_online(state: State):
    def extract_text_from_html(html_content: str) -> str:
        soup = BeautifulSoup(html_content, "html.parser")
        return soup.get_text()

    messages = state["messages"]
    search_query = state["search_query"]

    urls = list(
        google_search(search_query, region="jp", lang="jp", safe=True, num_results=1)
    )

    if len(urls) == 0:
        return {
            "messages": ToolMessage(
                content="",
                tool_call_id="",
            )
        }

    documents = []
    web_page_loader = AsyncChromiumLoader(
        urls=urls,
        user_agent="Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Mobile Safari/537.36",
    )
    documents.extend(web_page_loader.load())

    for document in documents:
        document.page_content = extract_text_from_html(document.page_content)

    splitted_documents = split_documents(
        documents=documents,
        separators=["。"],
        chunk_size=256,
        chunk_overlap=0,
    )

    embedding_model = fetch_embedding_model(
        base_url="http://ollama:11434",
        model_name="multilingual-e5-large",
    )

    semantic_retriever = construct_semantic_retriever(
        documents=splitted_documents,
        embedding_model=embedding_model,
        k=3,
    )
    retrieve_documents = semantic_retriever.invoke(search_query)
    for doc in retrieve_documents:
        messages.append(
            ToolMessage(
                content=doc.page_content,
                tool_call_id="",
            )
        )

    return {"messages": messages}


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
    graph.add_node("search_online", search_online)

    # add edge
    graph.add_edge(START, "generate_search_query")
    graph.add_edge("generate_search_query", "retrieve_documents")
    graph.add_conditional_edges(
        "retrieve_documents",
        check_search_online_requirement,
        {"search_online": "search_online", "llm_agent": "llm_agent"},
    )
    graph.add_edge("search_online", "llm_agent")
    graph.add_edge("llm_agent", END)

    # compile graph
    runner = graph.compile()

    question = "僕のヒーローアカデミアの第7期 第2クールオープニング曲は何ですか"
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
