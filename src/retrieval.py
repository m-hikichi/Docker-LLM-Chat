import langchain_core
from typing import Iterable, List, Optional
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from llamacpp import load_llamacpp_model


def load_embedding_model(
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    show_progress: bool = False,
) -> HuggingFaceEmbeddings:
    """
    Initialize the sentence_transformer.
    Args:
        model_name : Model name to use.
        show_progress : Whether to show a progress bar.
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        show_progress=show_progress,
    )


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

    semantic_retriever = construct_semantic_retriever(
        documents=splitted_documents,
        embedding_model=HuggingFaceEmbeddings(
            model_name="/workspace/models/multilingual-e5-large"
        ),
        k=3,
    )

    llm_model = load_llamacpp_model(
        model_path="/workspace/models/ELYZA-japanese-Llama-2-13b-fast-instruct-gguf/ELYZA-japanese-Llama-2-13b-fast-instruct-q5_K_M.gguf",
        n_ctx=2048,
        n_gpu_layers=-1,
        temperature=0.2,
        top_p=0.95,
        top_k=50,
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""<s>[INST] <<SYS>>
あなたは誠実で優秀な日本人のアシスタントです。以下の「コンテキスト情報」を元に「質問」に回答してください。
なおコンテキスト情報に無い情報は回答に含めないでください。
コンテキスト情報から回答が導けない場合は「わかりません」と回答してください。
<</SYS>>

# コンテキスト情報
{context}

# 質問
{question}[/INST]""",
    )

    chain = (
        {"context": semantic_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm_model
    )

    question = "イーブイからニンフィアに進化させる方法を教えてください。"
    for text in chain.stream(question):
        print(text, flush=True, end="")
