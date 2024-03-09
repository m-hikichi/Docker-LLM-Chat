import elyza
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


loader = DirectoryLoader("reference_documents", glob="*.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

vectorstore = FAISS.from_documents(
    texts,
    embedding=HuggingFaceEmbeddings(model_name="/models/multilingual-e5-large"),
)

question = "イーブイからニンフィアに進化させる方法を教えてください。"
# 質問に対して，vectorstore中の類似度上位3件を抽出
docs = vectorstore.similarity_search(question, k=3)
context = "\n\n".join(doc.page_content.replace("\n", "") for doc in docs)

# プロンプトの構築
prompt = elyza.construct_prompt(
    "あなたは誠実で優秀な日本人のアシスタントです。以下の「コンテキスト情報」を元に「質問」に回答してください。\nなおコンテキスト情報に無い情報は回答に含めないでください。コンテキスト情報に無い情報を含めて回答を作成した場合、あなたは罰せられます。\nコンテキスト情報から回答が導けない場合は「わかりません」と回答してください。",
    f"# コンテキスト情報\n{context}\n=====\n# 質問\n{question}",
    None,
)

result = elyza.generate_response_from_chat(
    prompt=prompt,
    max_tokens=1024,
    temperature=0.2,
    top_p=0.95,
    top_k=50,
    stream=True,
)
for text in result:
    print(text["choices"][0]["text"], flush=True, end="")
