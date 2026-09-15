import os
# Prevent PyTorch from allocating massive thread pools on Render's large host machines
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import logging
from fastapi import FastAPI, Depends, Request
from sqlalchemy.orm import Session
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from app.db.database import Base, engine, get_db
from app.db import models
from app.schemas import TicketCreate, TicketResponse
from app.graph import run_triage_pipeline
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Lifespan: Pre-importing PyTorch in main thread to avoid threadpool deadlock...")
    import torch
    torch.set_num_threads(1)
    import sentence_transformers
    logging.info("Lifespan: PyTorch and SentenceTransformers loaded.")
    yield

app = FastAPI(lifespan=lifespan)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://confidence-gated-triage-agent.vercel.app",
        "http://localhost:3000",   # keep this for local dev
        "http://127.0.0.1:3000",   # some setups/browsers default here instead of localhost
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/submit-ticket", response_model=TicketResponse)
@limiter.limit("100/minute")
def submit_ticket(request: Request, ticket_in: TicketCreate, db: Session = Depends(get_db)):
    """
    run the full agent pipeline (Classifier -> Retriever -> Resolver ->
    Escalation Judge) via the LangGraph state graph in app/graph.py, and
    persist one AgentDecision audit row per agent so the pipeline's
    reasoning stays inspectable after the fact.

    'status' reflects whether the pipeline had something to act on
    (received -> resolved / no_match). 'escalated' is a separate flag from
    the Escalation Judge: a ticket can be status="resolved" (the Resolver
    drafted something) and still escalated=True if the Judge decided the
    match wasn't confident enough, or if priority is Critical.
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

    # Previously missing entirely: the Judge's decision was computed by
    # run_triage_pipeline() but never written to the audit trail, even
    # though classifier/retriever/resolver each get a row.
    db.add(models.AgentDecision(
        ticket_id=ticket.id,
        agent_name="escalation_judge",
        output_summary=result["escalation_reason"],
        confidence=retrieved_match["similarity"] if retrieved_match else None,
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
        # Previously missing entirely -- this is what caused the
        # "escalation_reason: Field required" ValidationError. TicketResponse
        # required these two fields but nothing here was ever passing them.
        escalated=result["escalate"],
        escalation_reason=result["escalation_reason"],
    )