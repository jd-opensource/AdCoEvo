"""LangGraph control flow for one advertising period.

The callbacks own the existing auction, policy, and logging logic.  This graph
only orders their causal boundaries: observation, planning, one optimizer tick
at a time, and settlement inspection.
"""

from __future__ import annotations

from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph


class PeriodState(TypedDict):
    context: dict[str, Any]
    done: bool


PeriodNode = Callable[[PeriodState], dict[str, Any]]


def build_period_graph(
    warmup: PeriodNode,
    planner: PeriodNode,
    optimizer_step: PeriodNode,
    inspector: PeriodNode,
):
    """Compile one period's execution graph.

    ``optimizer_step`` handles exactly one environment tick and returns its
    ``done`` flag.  Invoke with ``{'context': ctx, 'done': False}`` and a
    recursion limit above the number of remaining ticks (128 covers 48).
    """
    graph = StateGraph(PeriodState)
    graph.add_node("warmup", warmup)
    graph.add_node("planner", planner)
    graph.add_node("optimizer_step", optimizer_step)
    graph.add_node("inspector", inspector)
    graph.add_edge(START, "warmup")
    graph.add_edge("warmup", "planner")
    graph.add_edge("planner", "optimizer_step")
    graph.add_conditional_edges(
        "optimizer_step",
        lambda state: "inspector" if state["done"] else "optimizer_step",
        {"optimizer_step": "optimizer_step", "inspector": "inspector"},
    )
    graph.add_edge("inspector", END)
    return graph.compile()
