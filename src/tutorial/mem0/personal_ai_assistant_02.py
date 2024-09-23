import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
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

class PersonalTravelAssistant:
    def __init__(self):
        self.client = ChatOpenAI(
            model="Llama-3.1-EZO:8B",
            temperature=0.1,
            max_tokens=1024,
        )
        self.memory = Memory.from_config(config)
        self.messages = [SystemMessage("あなたは誠実で優秀な日本人のアシスタントです。")]

    def ask_question(self, question, user_id):
        # Fetch previous related memories
        previous_memories = self.search_memories(question, user_id=user_id)
        if previous_memories:
            prompt = f"User input: {question}\n Previous memories: {previous_memories}"
        else:
            prompt = question
        self.messages.append(HumanMessage(prompt))

        prompt = ChatPromptTemplate.from_messages(self.messages)

        # construct chain
        chain = prompt | self.client

        # Generate response
        answer = ""
        for text in chain.stream({}):
            answer += text.content
        self.messages.append(AIMessage(answer))

        # Store the question in memory
        self.memory.add(question, user_id=user_id)
        return answer

    def get_memories(self, user_id):
        memories = self.memory.get_all(user_id=user_id)
        return [m['memory'] for m in memories['results']]

    def search_memories(self, query, user_id):
        memories = self.memory.search(query, user_id=user_id)
        return [m['memory'] for m in memories['results']]

# Usage example
user_id = "user_123"
ai_assistant = PersonalTravelAssistant()

def main():
    while True:
        question = input("Question: ")
        if question.lower() in ['q', 'exit']:
            print("Exiting...")
            break

        answer = ai_assistant.ask_question(question, user_id=user_id)
        print(f"Answer: {answer}")
        memories = ai_assistant.get_memories(user_id=user_id)
        print("Memories:")
        for memory in memories:
            print(f"- {memory}")
        print("-----")

if __name__ == "__main__":
    main()
