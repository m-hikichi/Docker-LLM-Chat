import elyza
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


vectorstore = FAISS.from_texts(
    [
        "ピカチュウは「かみなりのいし」を使うことで、ライチュウに進化します。",
        "ガーディは「ほのおのいし」を使うことで、ウインディに進化します。",
        "アローラロコンは「こおりのいし」を使うことで、アローラキュウコンに進化します。",
        "ヒンバスは「きれいなウロコ」を持たせて通信交換することで、ミロカロスに進化します。"
    ],
    embedding=HuggingFaceEmbeddings(model_name="/models/multilingual-e5-large")
)

question = "アローラロコンを進化させるにはどうすればいいですか"
# 質問に対して，vectorstore中の類似度上位1件を抽出
docs = vectorstore.similarity_search(question, k=1)

result = elyza.chat(
    message=f"# コンテキスト情報\n{docs[0].page_content}\n\n# 質問\n{question}",
    history=None,
    system_prompt="あなたは誠実で優秀な日本人のアシスタントです。以下の「コンテキスト情報」を元に「質問」に回答してください。\nなおコンテキスト情報に無い情報は回答に含めないでください。コンテキスト情報に無い情報を含めて回答を作成した場合、あなたは罰せられます。\nコンテキスト情報から回答が導けない場合は「わかりません」と回答してください。",
    max_tokens=512,
    temperature=0.2,
    top_p=0.95,
    top_k=50,
)
print(result)
