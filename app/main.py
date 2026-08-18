import logging

from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session

from app.db.database import Base, engine, get_db
from app.db import models
from app.schemas import TicketCreate, TicketResponse
from app.graph import run_triage_pipeline
from fastapi.middleware.cors import CORSMiddleware


logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="IT Support Triage Agent",
    description="Multi-agent system for ticket classification, retrieval, and confidence-gated escalation.",
    version="0.1.0",
)

#create tables directly from models. Once the schema stabilizes,
# switch to Alembic migrations instead of calling this on every startup.
Base.metadata.create_all(bind=engine)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://confidence-gated-triage-agent.vercel.app",
    ],
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
        agent_name="classifier",
        output_summary=f"Category: {result['category']}, Priority: {result['priority']}",
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
    )