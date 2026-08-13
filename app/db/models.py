import uuid
from datetime import datetime

from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

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
