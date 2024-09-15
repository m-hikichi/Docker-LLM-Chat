import os
from mem0 import Memory


config = {
    "llm": {
        "provider": "ollama",
        "config": {
            "ollama_base_url": "http://ollama:11434",
            "model": "ELYZA:8B-Q4_K_M",
            "temperature": 0.2,
            "max_tokens": 1500,
        }
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "ollama_base_url": "http://ollama:11434",
            "model": "multilingual-e5-large:latest",
        }
    },
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "test",
            "path": "db",
        }
    }
}

m = Memory.from_config(config)
result = m.add(
    "テニスのスキルアップに取り組んでいます。オンラインコースをいくつか紹介してください。",
    user_id="太郎",
    metadata={"カテゴリー": "趣味"}
)
print(result)
