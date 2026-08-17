"""
LangGraph state graph wiring the agents into a single
pipeline: Classifier -> Retriever -> Resolver -> Escalation Judge.

Kept as a thin orchestration layer over the existing agent functions --
each agent module (classifier.py, retriever.py, resolver.py,
escalation_judge.py) still owns its own logic and can be tested standalone
(see data/test_classifier.py and data/test_pipeline.py). This file only
owns the sequencing and the shared state that flows between nodes.

Week 3 added the fourth node -- the Escalation Judge -- as a final gate
after "resolve", deciding whether the pipeline's confidence is high enough
to auto-resolve or whether the ticket should go to a human instead.
"""
import logging
from typing import Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from sqlalchemy.orm import Session

from app.agents.classifier import classify_ticket
from app.agents.retriever import retrieve_resolution
from app.agents.resolver import resolve_ticket
from app.agents.escalation_judge import judge_escalation

logger = logging.getLogger(__name__)


class TriageState(TypedDict):
    # --- Input ---
    subject: str
    description: str
    db: Session
    # Not serialized anywhere -- the graph runs in-process within a single
    # FastAPI request, so passing the live SQLAlchemy session through state
    # is fine. The caller (main.py) still owns opening/closing it via the
    # usual get_db() dependency.

    # --- Filled in by classify_node ---
    category: Optional[str]
    priority: Optional[str]

    # --- Filled in by retrieve_node ---
    retrieved_match: Optional[dict]

    # --- Filled in by resolve_node ---
    resolution_status: Optional[str]  # "resolved" | "no_match"
    draft_resolution: Optional[str]
    tool_called: Optional[str]
    tool_result: Optional[str]

    # --- Filled in by judge_node ---
    escalate: Optional[bool]
    escalation_reason: Optional[str]


def classify_node(state: TriageState) -> dict:
    result = classify_ticket(state["subject"], state["description"])
    logger.info(f"[graph] classified as {result['category']} / {result['priority']}")
    return {"category": result["category"], "priority": result["priority"]}


def retrieve_node(state: TriageState) -> dict:
    # Retriever was built against short "issue summary"-style text, so
    # concatenate subject + description the same way the Week 2 seed
    # embeddings were generated, rather than passing them separately.
    ticket_text = f"{state['subject']}. {state['description']}"
    match = retrieve_resolution(state["db"], ticket_text)

    if match:
        logger.info(
            f"[graph] retrieved match '{match['matched_issue']}' "
            f"(similarity={match['similarity']})"
        )
    else:
        logger.info("[graph] no confident retrieval match")

    return {"retrieved_match": match}


def resolve_node(state: TriageState) -> dict:
    result = resolve_ticket(state["subject"], state["description"], state["retrieved_match"])
    return {
        "resolution_status": result["status"],
        "draft_resolution": result["draft_resolution"],
        "tool_called": result["tool_called"],
        "tool_result": result["tool_result"],
    }


def judge_node(state: TriageState) -> dict:
    result = judge_escalation(
        resolution_status=state["resolution_status"],
        retrieved_match=state["retrieved_match"],
        priority=state["priority"],
    )
    logger.info(f"[graph] escalate={result['escalate']} ({result['reason']})")
    return {"escalate": result["escalate"], "escalation_reason": result["reason"]}


def build_graph():
    graph = StateGraph(TriageState)
    graph.add_node("classify", classify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("resolve", resolve_node)
    graph.add_node("judge", judge_node)

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "resolve")
    graph.add_edge("resolve", "judge")
    graph.add_edge("judge", END)

    return graph.compile()


# Compiled once at import time, reused across requests -- same pattern as
# the Groq client (classifier.py, resolver.py) and the embedding model
# (retriever.py): expensive setup done once, not per-request.
triage_graph = build_graph()


def run_triage_pipeline(db: Session, subject: str, description: str) -> TriageState:
    """
    Runs the full Classifier -> Retriever -> Resolver pipeline for one
    ticket and returns the final state.
    """
    initial_state: TriageState = {
        "subject": subject,
        "description": description,
        "db": db,
        "category": None,
        "priority": None,
        "retrieved_match": None,
        "resolution_status": None,
        "draft_resolution": None,
        "tool_called": None,
        "tool_result": None,
        "escalate": None,
        "escalation_reason": None,
    }
    return triage_graph.invoke(initial_state)