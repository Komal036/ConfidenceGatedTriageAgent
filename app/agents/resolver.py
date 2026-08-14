import json
import logging

from groq import Groq

from app.config import settings
from app.tools.mock_tools import check_system_status, reset_password, lookup_account

logger = logging.getLogger(__name__)

client = Groq(api_key=settings.groq_api_key)

# Maps the tool name the LLM can choose (as a string) to the actual callable.
# Keeping this as an explicit whitelist — rather than letting the LLM invoke
# arbitrary functions — is deliberate: the Resolver can only ever trigger
# one of these three exact actions, nothing else.
AVAILABLE_TOOLS = {
    "check_system_status": check_system_status,
    "reset_password": reset_password,
    "lookup_account": lookup_account,
    "none": None,
}

RESOLVER_PROMPT = """You are a support resolution assistant. A ticket has been matched \
to a known issue in the knowledge base. Your job is to draft a helpful, specific \
resolution message for the user, and decide if calling a tool would help resolve it.

Ticket subject: {subject}
Ticket description: {description}

Matched known issue: {matched_issue}
Reference resolution: {resolution_text}

Available tools:
- "check_system_status": use if the issue might be a broader outage, not user-specific
- "reset_password": use ONLY if the ticket is specifically about a forgotten/locked password
- "lookup_account": use if you need to check the user's account status before resolving
- "none": use if no tool is needed — the reference resolution is enough to draft a reply

Output ONLY a JSON object with:
- "draft_resolution": a 2-3 sentence message to the user, written in a helpful, direct tone, \
based on the reference resolution but adapted to their specific wording
- "call_tool": one of "check_system_status", "reset_password", "lookup_account", "none"

Respond with ONLY the JSON object, no other text.
"""


def resolve_ticket(subject: str, description: str, retrieved_match: dict | None) -> dict:
    """
    Given a ticket and (optionally) a retrieved knowledge base match, drafts
    a resolution and decides whether to call a tool.

    If retrieved_match is None (Retriever found no confident match), this
    function doesn't attempt to draft a resolution at all — there's nothing
    reliable to base it on. This is a deliberate handoff point: a ticket
    with no match is exactly the kind of case the Escalation Judge (Week 3)
    should be looking at.

    Returns a dict: {
        "status": "resolved" | "no_match",
        "draft_resolution": str | None,
        "tool_called": str | None,
        "tool_result": str | None,
    }
    """
    if retrieved_match is None:
        return {
            "status": "no_match",
            "draft_resolution": None,
            "tool_called": None,
            "tool_result": None,
        }

    prompt = RESOLVER_PROMPT.format(
        subject=subject,
        description=description,
        matched_issue=retrieved_match["matched_issue"],
        resolution_text=retrieved_match["resolution_text"],
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # slightly higher than the classifier — drafting text benefits from a little variation
            max_tokens=300,
        )
        raw_output = response.choices[0].message.content.strip()
        if raw_output.startswith("```"):
            raw_output = raw_output.strip("`").removeprefix("json").strip()

        parsed = json.loads(raw_output)
        draft_resolution = parsed.get("draft_resolution", retrieved_match["resolution_text"])
        tool_name = parsed.get("call_tool", "none")

        if tool_name not in AVAILABLE_TOOLS:
            logger.warning(f"Resolver requested unknown tool '{tool_name}', ignoring.")
            tool_name = "none"

        tool_result = None
        tool_fn = AVAILABLE_TOOLS[tool_name]
        if tool_fn is not None:
            # .invoke() is how LangChain @tool-wrapped functions are called
            tool_result = tool_fn.invoke({list(tool_fn.args.keys())[0]: subject})

        return {
            "status": "resolved",
            "draft_resolution": draft_resolution,
            "tool_called": tool_name if tool_name != "none" else None,
            "tool_result": tool_result,
        }

    except (json.JSONDecodeError, KeyError, IndexError, AttributeError) as e:
        logger.error(f"Resolver failed to parse LLM output: {e}")
        # Fail safe: fall back to the raw knowledge base resolution text,
        # with no tool call — same "fail safe, not silent" pattern as the classifier.
        return {
            "status": "resolved",
            "draft_resolution": retrieved_match["resolution_text"],
            "tool_called": None,
            "tool_result": None,
        }
