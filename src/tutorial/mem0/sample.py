import os
import json
from mem0 import Memory


config = {
    "llm": {
        "provider": "ollama",
        "config": {
            "ollama_base_url": "http://ollama:11434",
            "model": "Llama-3.1-EZO:8B",
            "temperature": 0.2,
            "max_tokens": 1500,
        }
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "ollama_base_url": "http://ollama:11434",
            "model": "mxbai-embed-large",
        }
    },
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "test",
            "path": "db",
        }
    },
    "version": "v1.1"
}

# メモリの初期化
m = Memory.from_config(config)
print(json.dumps(m.get_all(), indent=2, ensure_ascii=False))

# メモリに追加
result = m.add(
    "テニスのスキルアップに取り組んでいます。オンラインコースをいくつか紹介してください。",
    user_id="太郎",
    metadata={"カテゴリー": "趣味"}
)
print(result)
print(json.dumps(m.get_all(), indent=2, ensure_ascii=False))

# 検索
related_memories = m.search(query="太郎の趣味は？", user_id="太郎")
print(json.dumps(related_memories, indent=2, ensure_ascii=False))