"""Lai: planner định hướng, ReAct thực hiện; graph quyết định khi nào replan."""

from langgraph.graph import END, START, StateGraph

from harness import Harness
from react_agent import invoke_react
from workflow import WorkflowNodes, WorkflowState


def build_hybrid_graph(nodes: WorkflowNodes):
    def execute(state: WorkflowState) -> dict:
        result = invoke_react(nodes.harness, nodes.backend, nodes.model_name, state["plan"])
        return {"answer": result["answer"], "error": result["error"],
                "last_status": result["last_status"]}

    def after_verify(state: WorkflowState) -> str:
        # Chỉ tìm lại/replan khi hết chỗ; lỗi quyền/API/budget cần bàn giao.
        if (not state["done"] and state.get("last_status") == "unavailable"
                and state["attempts"] < nodes.max_plans):
            nodes.harness.record("replan", reason=state["error"])
            return "search"
        return "finalize"

    graph = StateGraph(WorkflowState)
    graph.add_node("search", nodes.search)
    graph.add_node("plan", nodes.plan)
    graph.add_node("review", nodes.review)
    graph.add_node("execute", execute)
    graph.add_node("verify", nodes.verify)
    graph.add_node("finalize", nodes.finalize)
    graph.add_edge(START, "search")
    graph.add_conditional_edges("search", nodes.after_search, ["plan", "finalize"])
    graph.add_conditional_edges("plan", nodes.after_plan, ["review", "finalize"])
    graph.add_conditional_edges("review", nodes.after_review, ["execute", "plan", "finalize"])
    graph.add_edge("execute", "verify")
    graph.add_conditional_edges("verify", after_verify, ["search", "finalize"])
    graph.add_edge("finalize", END)
    return graph.compile()


def run_hybrid(harness: Harness, backend="mock", model_name=None, reviewer=None) -> dict:
    nodes = WorkflowNodes(harness, "hybrid", backend, model_name, reviewer)
    state = build_hybrid_graph(nodes).invoke({"attempts": 0}, config={"recursion_limit": 40})
    return state["output"]
