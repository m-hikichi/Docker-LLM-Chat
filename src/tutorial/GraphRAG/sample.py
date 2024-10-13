import os
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_experimental.llms.ollama_functions import OllamaFunctions


# neo4j db settings
os.environ["OPENAI_API_KEY"] = "dummpy-api-key"
os.environ["NEO4J_URI"] = "bolt://neo4j:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "password"
graph = Neo4jGraph()

# prepare the text
loader = TextLoader(file_path="ゆりかごの星.txt")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=128, chunk_overlap=16)
documents = text_splitter.split_documents(documents=docs)

# LLM settings & graph construction
llm = OllamaFunctions(
    base_url="http://ollama:11434",
    model="Llama-3.1-EZO:8B",
    temperature=0,
    format="json",
)
# llm = ChatOpenAI(model="Llama-3.1-EZO:8B", temperature=0)
llm_transformer = LLMGraphTransformer(llm=llm)
graph_documents = llm_transformer.convert_to_graph_documents(documents)

# upload graphed-documents to neo4j
graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)
