from langgraph.graph import Graph


# Node の定義
def node_a(input):
    input += "Hello "
    return input

def node_b(input):
    input += "World!"
    return input

# Graph の作成
graph = Graph()

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
# 作成した Graph を画像で保存
with open("graph.png", "wb") as png:
    png.write(runner.get_graph().draw_mermaid_png())

# 実行
runner.invoke("LangGraph: ", debug=True)
