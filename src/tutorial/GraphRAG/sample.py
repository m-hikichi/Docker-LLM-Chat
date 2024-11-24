# reference: https://blog.langchain.dev/enhancing-rag-based-applications-accuracy-by-constructing-and-leveraging-knowledge-graphs/
# reference: https://github.com/Coding-Crashkurse/GraphRAG-with-Llama-3.1/blob/main/enhancing_rag_with_graph.ipynb

import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_ollama import OllamaEmbeddings
from pydantic import BaseModel, Field

# neo4j db settings
os.environ["OPENAI_API_KEY"] = "dummpy-api-key"
os.environ["NEO4J_URI"] = "bolt://neo4j:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "password"
graph = Neo4jGraph()

# prepare the text
loader = TextLoader(file_path="ゆりかごの星.txt")
docs = loader.load()
# Define chunking strategy
text_splitter = RecursiveCharacterTextSplitter(chunk_size=128, chunk_overlap=16)
documents = text_splitter.split_documents(documents=docs)

# LLM settings & graph construction
llm = OllamaFunctions(
    base_url="http://ollama:11434",
    model="Llama-3.1-EZO:8B",
    temperature=0,
    format="json",
)
llm_transformer = LLMGraphTransformer(llm=llm)
# Extract graph data
graph_documents = llm_transformer.convert_to_graph_documents(documents)

# Store to neo4j
graph.add_graph_documents(graph_documents, baseEntityLabel=True, include_source=True)

# Hybrid Retrieval for RAG
embeddings = OllamaEmbeddings(
    base_url="http://ollama:11434",
    model="bge-m3",
)
vector_index = Neo4jVector.from_existing_graph(
    embeddings,
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding",
)
vector_retriever = vector_index.as_retriever()


# Extract entities from text
class Entities(BaseModel):
    """Identifying information about entities."""

    names: list[str] = Field(
        ...,
        description="All the person, organization, or business entities that "
        "appear in the text",
    )


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting organization and person entities from the text.",
        ),
        (
            "human",
            "Use the given format to extract information from the following"
            "input: {question}",
        ),
    ]
)

entity_chain = prompt | llm.with_structured_output(Entities)
print(entity_chain.invoke({"question": "スレッタのお母さんは何をしている人ですか？"}))

# define a full-text index and a function that will generate full-text queries that allow a bit of misspelling
graph.query(
    "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]"
)


def generate_full_text_query(input: str) -> str:
    """
    Generate a full-text search query for a given input string.

    This function constructs a query string suitable for a full-text
    search. It processes the input string by splitting it into words and
    appending a similarity threshold (~2 changed characters) to each
    word, then combines them using the AND operator. Useful for mapping
    entities from user questions to database values, and allows for some
    misspelings.
    """
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()


# Fulltext index query
def structured_retriever(question: str) -> str:
    """
    Collects the neighborhood of entities mentioned
    in the question
    """
    result = ""
    entities = entity_chain.invoke(question)
    for entity in entities.names:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('fulltext_entity_id', $query, {limit:2})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": entity},
        )
        result += "\n".join([el["output"] for el in response])
    return result


print(structured_retriever("スレッタのお母さんは何をしている人ですか？"))


# final retriever
def full_retriever(question: str):
    graph_data = structured_retriever(question)
    vector_data = [el.page_content for el in vector_index.similarity_search(question)]
    final_data = f"""Structured data:
{graph_data}
Unstructured data:
{"#Document ". join(vector_data)}
    """
    return final_data


# defining the RAG chain
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {
        "context": full_retriever,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

chain.invoke("スレッタのお母さんは何をしている人ですか？")
