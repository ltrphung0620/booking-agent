"""Plan-then-Execute: LangChain planner, LangGraph điều phối, code chạy plan."""

from langgraph.graph import END, START, StateGraph

from harness import Harness
from workflow import WorkflowNodes, WorkflowState


def build_plan_graph(nodes: WorkflowNodes):
    graph = StateGraph(WorkflowState)
    graph.add_node("search", nodes.search)
    graph.add_node("plan", nodes.plan)
    graph.add_node("review", nodes.review)
    graph.add_node("execute", nodes.execute_plan)
    graph.add_node("finalize", nodes.finalize)
    graph.add_edge(START, "search")
    graph.add_conditional_edges("search", nodes.after_search, ["plan", "finalize"])
    graph.add_conditional_edges("plan", nodes.after_plan, ["review", "finalize"])
    graph.add_conditional_edges("review", nodes.after_review, ["execute", "plan", "finalize"])
    # Executor gặp lỗi sẽ dừng; mẫu này không tự lập lại plan sau execution.
    graph.add_edge("execute", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def run_plan(harness: Harness, backend="mock", model_name=None, reviewer=None) -> dict:
    nodes = WorkflowNodes(harness, "plan", backend, model_name, reviewer)
    state = build_plan_graph(nodes).invoke({"attempts": 0}, config={"recursion_limit": 30})
    return state["output"]
