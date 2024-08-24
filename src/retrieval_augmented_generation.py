from pathlib import Path
from typing import Iterable, List, Optional

import langchain_core
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever

from llms.llm_api import fetch_llm_api_model
from embeddings.huggingface_embedding import load_embedding_model


def load_documents(
    path: str,
) -> List[langchain_core.documents.Document]:
    """
    Args:
        path : Path to directory.
    """
    documents = []
    filepath_list = Path(path).glob("?*.*")

    # load documents
    for filepath in filepath_list:
        if filepath.suffix == ".txt":
            text_loader = TextLoader(str(filepath))
            documents.extend(text_loader.load())
        else:
            print(f"{filepath} has been removed from RAG reference documents.")

    return documents


def split_documents(
    documents: Iterable[langchain_core.documents.Document],
    separators: Optional[List[str]] = ["\n\n", "\n", " ", ""],
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
) -> List[langchain_core.documents.Document]:
    """
    Args:
        documents :
        separators :
        chunk_size : Maximum size of chunks to return
        chunk_overlap : Overlap in characters between chunks
    """
    # initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # split documents
    return text_splitter.split_documents(documents)


def construct_semantic_retriever(
    documents: List[langchain_core.documents.Document],
    embedding_model: langchain_core.embeddings.Embeddings,
    k: int = 4,
    score_threshold: float = 0.0,
) -> VectorStoreRetriever:
    """
    Args:
        documents :
        embedding_model :
        k : Amount of documents to return
        score_threshold : Minimum relevance threshold for similarity_score_threshold
    """
    #
    vectorstore = FAISS.from_documents(documents, embedding_model)

    #
    semantic_retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": score_threshold},
    )
    return semantic_retriever


def format_docs(docs: List[langchain_core.documents.Document]) -> str:
    return f"\n{'-'*100}\n".join(
        [f"Document {i+1}:\n\n" + doc.page_content for i, doc in enumerate(docs)]
    )


if __name__ == "__main__":
    documents = load_documents("/workspace/documents")

    splitted_documents = split_documents(
        documents=documents,
        separators=["。"],
        chunk_size=256,
        chunk_overlap=0,
    )

    embedding_model = load_embedding_model(
        model_path="/workspace/models/multilingual-e5-large",
    )

    semantic_retriever = construct_semantic_retriever(
        documents=splitted_documents,
        embedding_model=embedding_model,
        k=3,
    )

    llm_model = fetch_llm_api_model(
        model="ELYZA:8B-Q4_K_M",
        temperature=0.2,
        max_tokens=2048,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                """あなたは誠実で優秀な日本人のアシスタントです。以下の「コンテキスト情報」を元に「質問」に回答してください。
なおコンテキスト情報に無い情報は回答に含めないでください。
コンテキスト情報から回答が導けない場合は「わかりません」と回答してください。"""
            ),
            HumanMessagePromptTemplate.from_template("# コンテキスト情報\n{context}"),
            HumanMessagePromptTemplate.from_template("# 質問\n{question}"),
        ]
    )

    chain = (
        {"context": semantic_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm_model
    )

    question = "イーブイからニンフィアに進化させる方法を教えてください。"
    for text in chain.stream(question):
        print(text.content, flush=True, end="")
    print()
