"""
Mock tool APIs. These simulate real enterprise systems (a real ticketing
platform would call actual Zendesk/Freshdesk/internal APIs here) — the
point of this project is demonstrating tool-calling patterns and
confidence-gated decision-making, not building real IT infrastructure.

Each function is wrapped with @tool so LangChain/LangGraph can expose it
to the Resolver Agent as something the LLM can choose to call.
"""
import random
from langchain_core.tools import tool


@tool
def check_system_status(service_name: str) -> str:
    """Check whether a given service or system is currently experiencing an outage."""
    # Simulated — in a real system this would hit a status API (e.g. Statuspage).
    known_issues = {"network": False, "email": False, "billing": False}
    is_down = known_issues.get(service_name.lower(), False)
    if is_down:
        return f"{service_name} is currently experiencing a known outage. No user-side fix needed."
    return f"{service_name} is operating normally. No known outages."


@tool
def reset_password(user_email: str) -> str:
    """Trigger a password reset email for the given user account."""
    # Simulated — a real implementation would call an auth provider's API.
    return f"Password reset email sent to {user_email}. Link expires in 30 minutes."


@tool
def lookup_account(user_email: str) -> str:
    """Look up basic account status (active, locked, suspended) for a user."""
    # Simulated with a fixed response for demo purposes — a real implementation
    # would query the actual user database.
    statuses = ["active", "locked (too many failed logins)", "active — no recent issues"]
    return f"Account for {user_email}: {random.choice(statuses)}"


ALL_TOOLS = [check_system_status, reset_password, lookup_account]
