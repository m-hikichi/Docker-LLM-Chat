from langgraph.graph import MessageGraph
from langchain_core.messages import HumanMessage


# Node の定義
def node_a(input):
    input[-1].content += "Hello "
    return input

def node_b(input):
    input[-1].content += "World!"
    return input

# Graph の作成
graph = MessageGraph()

# 定義した Node の追加
graph.add_node("node_a", node_a)
graph.add_node("node_b", node_b)

# entry_point と finish_point の定義
graph.set_entry_point("node_a")
graph.set_finish_point("node_b")
# edgeの追加（node_a 完了後 node_b を実行する設定）
graph.add_edge("node_a", "node_b")

# Graph のコンパイル
runner = graph.compile()

# 実行
runner.invoke(HumanMessage(content="LangGraph: "), debug=True)