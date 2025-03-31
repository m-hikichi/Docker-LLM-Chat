import requests
from typing import Optional
from langchain_ollama import OllamaEmbeddings


class EmbeddingAPIException(Exception):
    def __init__(self, api_url: str, response_status_code: Optional[int] = None):
        if response_status_code is None:
            self.message = f"Embedding API {api_url} is not available."
        else:
            self.message = f"Embedding API {api_url} is not available. Status code: {response_status_code}"

    def __str__(self):
        return self.message


def fetch_embedding_model(
    base_url: str = "http://localhost:11434",
    model_name: str = "llama2",
) -> OllamaEmbeddings:
    """
    Args:
        base_url : Base url the model is hosted under.
        model_name : Model name to use.
    Returns:
        OllamaEmbedding :
    """
    try:
        response = requests.get(base_url + "/v1/models")
        if response.status_code == 200:
            embedding_model = OllamaEmbeddings(
                base_url=base_url,
                model=model_name,
            )

            return embedding_model
        else:
            raise EmbeddingAPIException(base_url, response.status_code)

    except requests.exceptions.RequestException:
        raise EmbeddingAPIException(base_url)


if __name__ == "__main__":
    # load embedding model
    embedding_model = fetch_embedding_model(
        base_url="http://ollama:11434",
        model_name="multilingual-e5-large",
    )

    embeddings = embedding_model.embed_query("こんにちは")
    print(len(embeddings))
    print(embeddings[:5])
