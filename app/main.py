import logging

from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session

from app.db.database import Base, engine, get_db
from app.db import models
from app.schemas import TicketCreate, TicketResponse
from app.graph import run_triage_pipeline
from fastapi.middleware.cors import CORSMiddleware


logging.basicConfig(level=logging.INFO)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://confidence-gated-triage-agent.vercel.app",
        "http://localhost:3000",  # keep this for local dev
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/submit-ticket", response_model=TicketResponse)
def submit_ticket(ticket_in: TicketCreate, db: Session = Depends(get_db)):
    """
    run the full agent pipeline (Classifier -> Retriever ->
    Resolver) via the LangGraph state graph in app/graph.py, and persist one
    AgentDecision audit row per agent so the pipeline's reasoning stays
    inspectable after the fact.

    'status' progresses received -> resolved / no_match here. Week 3 adds
    the Escalation Judge, which will turn "no_match" (and low-confidence
    "resolved" cases) into an actual "escalated" state instead.
    """
    ticket = models.Ticket(
        subject=ticket_in.subject,
        description=ticket_in.description,
        product=ticket_in.product,
        channel=ticket_in.channel,
        status="received",
    )
    db.add(ticket)
    db.commit()
    db.refresh(ticket)

    result = run_triage_pipeline(db, ticket_in.subject, ticket_in.description)

    ticket.category = result["category"]
    ticket.priority = result["priority"]
    ticket.status = "resolved" if result["resolution_status"] == "resolved" else "no_match"
    db.commit()
    db.refresh(ticket)

    db.add(models.AgentDecision(
    ticket_id=ticket.id,
    agent_name="escalation_judge",
    output_summary=result["escalation_reason"],
    confidence=result["retrieved_match"]["similarity"] if result["retrieved_match"] else None,
))

    retrieved_match = result["retrieved_match"]
    if retrieved_match:
        db.add(models.AgentDecision(
            ticket_id=ticket.id,
            agent_name="retriever",
            output_summary=f"Matched: {retrieved_match['matched_issue']}",
            confidence=retrieved_match["similarity"],
        ))
    else:
        db.add(models.AgentDecision(
            ticket_id=ticket.id,
            agent_name="retriever",
            output_summary="No confident match found in knowledge base.",
        ))

    if result["resolution_status"] == "resolved":
        tool_note = f", tool called: {result['tool_called']}" if result["tool_called"] else ", no tool needed"
        db.add(models.AgentDecision(
            ticket_id=ticket.id,
            agent_name="resolver",
            output_summary=f"Drafted resolution{tool_note}.",
        ))

    db.commit()

    return TicketResponse(
    id=str(ticket.id),
    subject=ticket.subject,
    description=ticket.description,
    category=ticket.category,
    priority=ticket.priority,
    status=ticket.status,
    matched_issue=retrieved_match["matched_issue"] if retrieved_match else None,
    match_similarity=retrieved_match["similarity"] if retrieved_match else None,
    draft_resolution=result["draft_resolution"],
    tool_called=result["tool_called"],
    escalated=result["escalate"],
    escalation_reason=result["escalation_reason"],
)