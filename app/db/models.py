import uuid
from datetime import datetime

from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from app.db.database import Base


class Ticket(Base):
    """
    Represents one incoming support ticket, from raw submission through
    to the agent pipeline's final decision.

    'status' progresses: received -> classified -> retrieved -> resolved / escalated
    This lets you query "how far did this ticket get" if something fails mid-pipeline.
    """
    __tablename__ = "tickets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    subject = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    product = Column(String, nullable=True)
    channel = Column(String, nullable=True)

    # Filled in by the Classifier Agent (Week 1)
    category = Column(String, nullable=True)
    priority = Column(String, nullable=True)

    # Filled in later, by the Escalation Judge (Week 3)
    status = Column(String, default="received")  # received, classified, resolved, escalated
    final_confidence = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    decisions = relationship("AgentDecision", back_populates="ticket", cascade="all, delete-orphan")


class AgentDecision(Base):
    """
    An audit log entry: one row per agent, per ticket, recording what that
    agent decided and why. This is what makes the pipeline's reasoning
    inspectable later — and it's exactly what your eval harness in Week 3
    will read from.
    """
    __tablename__ = "agent_decisions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ticket_id = Column(UUID(as_uuid=True), ForeignKey("tickets.id"), nullable=False)

    agent_name = Column(String, nullable=False)  # "classifier", "retriever", "resolver", "escalation_judge"
    output_summary = Column(Text, nullable=False)  # human-readable summary of what the agent decided
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    ticket = relationship("Ticket", back_populates="decisions")


class Resolution(Base):
    """
    The knowledge base: one row per known issue + its resolution.

    'embedding' is a 384-dimensional vector (matching all-MiniLM-L6-v2's
    output size) generated from `issue_summary`. The Retriever Agent
    searches this table by comparing a new ticket's embedding against
    every row's embedding using cosine distance.

    Seeded by data/seed_knowledge_base.py — a mix of hand-written entries
    (since the Kaggle dataset's Resolution field is sparse) and any real
    resolutions pulled from the dataset that are usable as-is.
    """
    __tablename__ = "resolutions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    category = Column(String, nullable=False)
    issue_summary = Column(Text, nullable=False)     # short description of the problem
    resolution_text = Column(Text, nullable=False)   # the known fix
    embedding = Column(Vector(384), nullable=True)   # filled in by seed_knowledge_base.py

    created_at = Column(DateTime, default=datetime.utcnow)
