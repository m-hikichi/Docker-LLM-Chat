from langchain_huggingface import HuggingFaceEmbeddings


def load_embedding_model(
    model_path: str = "sentence-transformers/all-mpnet-base-v2",
    show_progress: bool = False,
) -> HuggingFaceEmbeddings:
    """
    Initialize the sentence_transformer.
    Args:
        model_path : Model path to use.
        show_progress : Whether to show a progress bar.
    """
    return HuggingFaceEmbeddings(
        model_name=model_path,
        show_progress=show_progress,
    )


if __name__ == "__main__":
    # load embedding model
    embedding_model = load_embedding_model(
        model_path="/workspace/models/multilingual-e5-large",
    )

    embeddings = embedding_model.embed_query("こんにちは")
    print(len(embeddings))
    print(embeddings[:5])
