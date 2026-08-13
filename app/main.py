import logging

from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session

from app.db.database import Base, engine, get_db
from app.db import models
from app.schemas import TicketCreate, TicketResponse
from app.agents.classifier import classify_ticket

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="IT Support Triage Agent",
    description="Multi-agent system for ticket classification, retrieval, and confidence-gated escalation.",
    version="0.1.0",
)

# Week 1: create tables directly from models. Once the schema stabilizes,
# switch to Alembic migrations instead of calling this on every startup.
Base.metadata.create_all(bind=engine)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/submit-ticket", response_model=TicketResponse)
def submit_ticket(ticket_in: TicketCreate, db: Session = Depends(get_db)):
    """
    Week 1 scope: store the ticket, run the Classifier Agent, save its output.

    In Week 2 this same function will grow to call the Retriever and Resolver
    agents in sequence via a LangGraph state graph. Keeping it a plain
    function call for now (rather than jumping straight to LangGraph) is
    intentional — get one agent working end-to-end before adding orchestration.
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

    classification = classify_ticket(ticket_in.subject, ticket_in.description)
    ticket.category = classification["category"]
    ticket.priority = classification["priority"]
    ticket.status = "classified"
    db.commit()
    db.refresh(ticket)

    decision = models.AgentDecision(
        ticket_id=ticket.id,
        agent_name="classifier",
        output_summary=f"Category: {classification['category']}, Priority: {classification['priority']}",
    )
    db.add(decision)
    db.commit()

    return TicketResponse(
        id=str(ticket.id),
        subject=ticket.subject,
        description=ticket.description,
        category=ticket.category,
        priority=ticket.priority,
        status=ticket.status,
    )
