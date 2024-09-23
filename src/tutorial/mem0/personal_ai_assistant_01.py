from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from mem0 import Memory

# Initialize the OpenAI client
client = ChatOpenAI(
    model="Llama-3.1-EZO:8B",
    temperature=0.1,
    max_tokens=1024,
)

class PersonalAITutor:
    def __init__(self):
        """
        Initialize the PersonalAITutor with memory configuration and OpenAI client.
        """
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
        self.memory = Memory.from_config(config)
        self.client = client
        self.app_id = "app-1"

    def ask(self, question, user_id=None):
        """
        Ask a question to the AI and store the relevant facts in memory

        :param question: The question to ask the AI.
        :param user_id: Optional user ID to associate with the memory.
        """
        # create prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage("あなたは誠実で優秀な日本人のアシスタントです。"),
                HumanMessagePromptTemplate.from_template("{query}"),
            ]
        )

        # construct chain
        chain = prompt | client

        # Store the question in memory
        self.memory.add(question, user_id=user_id, metadata={"app_id": self.app_id})

        # inference
        for text in chain.stream({"query": question}):
            print(text.content, flush=True, end="")
        print()

    def get_memories(self, user_id=None):
        """
        Retrieve all memories associated with the given user ID.

        :param user_id: Optional user ID to filter memories.
        :return: List of memories.
        """
        return self.memory.get_all(user_id=user_id)

# Instantiate the PersonalAITutor
ai_tutor = PersonalAITutor()

# Define a user ID
user_id = "john_doe"

# Ask a question
ai_tutor.ask("私の好きな食べ物はリンゴです", user_id=user_id)

# Fetching Memories
memories = ai_tutor.get_memories(user_id=user_id)
print(memories)