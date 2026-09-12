import json
import logging

from groq import Groq

from app.config import settings

logger = logging.getLogger(__name__)

client = Groq(api_key=settings.groq_api_key)

# Keep this list tight and matched to your dataset's real categories once
# you've explored the Kaggle CSV (Week 1, Day 1). A model does better with
# a fixed small set of options than an open-ended "pick anything" prompt.
VALID_CATEGORIES = [
    "Hardware", "Software", "Network", "Account Access",
    "Billing", "Data Loss", "General Inquiry"
]
VALID_PRIORITIES = ["Low", "Medium", "High", "Critical"]

CLASSIFIER_PROMPT = """You are a support ticket classifier. Given a ticket's subject \
and description, output ONLY a JSON object with two fields:

- "category": one of {categories}
- "priority": one of {priorities}

CATEGORY GUIDANCE:
- "General Inquiry" is for requests, questions, or feature suggestions where NOTHING is \
broken — e.g. "how do I upgrade my plan", "can you add dark mode". Do not classify these \
as Billing or Software just because they mention a product area. Only use Billing/Software/etc. \
when something is actually malfunctioning, disputed, or blocking the user.

PRIORITY GUIDANCE — pick based on actual impact, not just tone:
- "Critical": user is completely blocked (cannot log in at all, data is being actively lost, security breach)
- "High": significant disruption to the user's work with no workaround (locked account, repeated \
call drops, missing 2FA codes preventing login)
- "Medium": inconvenient or annoying, but the user can still get their work done (billing disputes, \
minor hardware issues, occasional glitches)
- "Low": cosmetic, informational, or a request/question with no urgency (feature requests, \
plan questions, minor invoice discrepancies)
Do not default to "Medium" as a safe middle ground — actively check whether the ticket describes \
a full blocker (favor High/Critical) or a non-issue request (favor Low) before settling on Medium.

EXAMPLES:
Subject: "My laptop keeps disconnecting from WiFi"
Description: "It drops every 10 minutes and I have to manually reconnect. Very annoying."
Output: {{"category": "Network", "priority": "Medium"}}

Subject: "Cannot login, getting 500 error on production"
Description: "No one in my team can access the system. It's completely down."
Output: {{"category": "Software", "priority": "Critical"}}

Subject: "Need help upgrading to the premium tier"
Description: "I want to purchase the premium plan but don't see the option."
Output: {{"category": "General Inquiry", "priority": "Low"}}

Subject: "Monitor sometimes flickers"
Description: "My external monitor flickers once a day for a second."
Output: {{"category": "Hardware", "priority": "Low"}}

Subject: "Locked out of my account, urgent"
Description: "I need to file taxes today and my password reset email is not arriving."
Output: {{"category": "Account Access", "priority": "High"}}

Ticket subject: {subject}
Ticket description: {description}

Respond with ONLY the JSON object, no other text.
"""


def classify_ticket(subject: str, description: str) -> dict:
    """
    Runs the Classifier Agent on a single ticket.

    Returns a dict: {"category": str, "priority": str, "raw_confidence": float}

    Falls back to a safe default ("General Inquiry" / "Medium") if the LLM
    output can't be parsed — this is deliberate: a classification failure
    should degrade gracefully, not crash the pipeline. This same "fail safe,
    not silent" instinct is what your Escalation Judge will formalize in Week 3.
    """
    prompt = CLASSIFIER_PROMPT.format(
        categories=", ".join(VALID_CATEGORIES),
        priorities=", ".join(VALID_PRIORITIES),
        subject=subject,
        description=description,
    )

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300,  # bumped from 100
            response_format={"type": "json_object"},
            reasoning_effort="low",  
        )
        raw_output = response.choices[0].message.content.strip()

        # Models sometimes wrap JSON in ```json fences despite instructions — strip if present
        if raw_output.startswith("```"):
            raw_output = raw_output.strip("`").removeprefix("json").strip()

        parsed = json.loads(raw_output)

        category = parsed.get("category", "General Inquiry")
        priority = parsed.get("priority", "Medium")

        if category not in VALID_CATEGORIES:
            logger.warning(f"Classifier returned unknown category '{category}', defaulting.")
            category = "General Inquiry"
        if priority not in VALID_PRIORITIES:
            logger.warning(f"Classifier returned unknown priority '{priority}', defaulting.")
            priority = "Medium"

        return {"category": category, "priority": priority}

    except (json.JSONDecodeError, KeyError, IndexError, AttributeError) as e:
        logger.error(f"Classifier failed to parse LLM output: {e}")
        return {"category": "General Inquiry", "priority": "Medium"}
