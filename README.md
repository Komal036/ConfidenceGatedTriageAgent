# 🎧 Autonomous IT Support Triage & Resolution Agent

### Multi-Agent System for Confidence-Gated Ticket Classification, Retrieval, and Escalation

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?style=flat-square&logo=fastapi)
![LangGraph](https://img.shields.io/badge/LangGraph-orchestration-purple?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-orange?style=flat-square)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-blue?style=flat-square&logo=postgresql)
![Dataset](https://img.shields.io/badge/Dataset-8.4k_tickets-yellow?style=flat-square)

<!-- Once you have a final eval number, add a badge like the SMAPE one in the reference repo, e.g.: -->
<!-- ![Escalation Precision](https://img.shields.io/badge/Escalation_Precision-XX%25-brightgreen?style=flat-square) -->

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Architecture](#️-architecture)
- [Tech Stack](#️-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#️-installation)
- [Usage](#-usage)
- [Agent Design Deep-Dive](#-agent-design-deep-dive)
- [Escalation Logic](#-escalation-logic)
- [Evaluation Strategy](#-evaluation-strategy)
- [Results](#-results)
- [Key Design Decisions](#-key-design-decisions)
- [Future Improvements](#-future-improvements)
- [Metrics Explained](#-metrics-explained)
- [References](#-references)

---

## 🎯 Overview

This project is a multi-agent system that automates first-line IT support ticket handling. Given an incoming ticket (subject, description, product/category context), the system classifies it, checks for a known resolution, attempts an autonomous fix via tool calls, and critically **knows when not to act**, escalating to a human agent when its own confidence is low.

**Final Result: [X]% resolution accuracy, [Y]% false-escalation rate, [Z]% false-confidence rate on a [N]-ticket held-out eval set.**


---

## 📋 Problem Statement

| Property | Details |
|---|---|
| Task | Multi-class classification + retrieval + gated autonomous action |
| Input | Ticket subject, description, product/category metadata |
| Output | Category, priority, resolution action OR escalation with reasoning |
| Metric | Resolution accuracy, false-escalation rate, false-confidence rate |
| Dataset | [8,469 tickets — Kaggle Customer Support Ticket Dataset] |
| Challenge | Balancing autonomy against the cost of being confidently wrong |

### Why is this hard?

- A ticket classifier alone is easy — the hard part is deciding **when the system shouldn't trust itself**
- Support tickets are noisy: vague subjects, missing detail, overlapping categories
- False resolution (confidently closing a ticket incorrectly) is more costly than false escalation (unnecessarily routing to a human) — the system needs asymmetric risk handling, not just accuracy
- No single "correct" escalation threshold exists — it's a tunable tradeoff you have to justify with data, not intuition
- Real support knowledge bases are incomplete; the Retriever agent must handle "no good match found" gracefully instead of hallucinating a fix

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────┐
│                    TICKET SUBMITTED                    │
│         subject · description · product · channel      │
└───────────────────────┬────────────────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   Classifier Agent   │
              │  category, priority  │
              └──────────┬───────────┘
                         │
                         ▼
              ┌─────────────────────┐        ┌──────────────────┐
              │   Retriever Agent    │◄──────►│  Knowledge Base    │
              │ known-issue matching │        │  (pgvector store)  │
              └──────────┬───────────┘        └──────────────────┘
                         │
                         ▼
              ┌─────────────────────┐        ┌──────────────────┐
              │   Resolver Agent     │◄──────►│    Tool APIs       │
              │  drafts fix, calls   │        │ status / reset /   │
              │  tools if confident  │        │ account lookup     │
              └──────────┬───────────┘        └──────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Escalation Judge    │
              │ confidence threshold │
              └──────────┬───────────┘
                    ┌─────┴─────┐
                    ▼           ▼
            ┌───────────┐ ┌─────────────────┐
            │Auto-       │ │ Escalated to     │
            │resolved    │ │ human (+reason)  │
            └───────────┘ └─────────────────┘
```

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Backend | FastAPI 0.115, Uvicorn, Pydantic v2, SQLAlchemy 2.0 |
| Database | PostgreSQL (Neon, pooled connection) |
| Vector store | pgvector |
| Agent orchestration | LangGraph, LangChain |
| LLM inference | Groq API — Llama 3.3 70B (`groq==1.6.0`) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Evaluation | pandas, custom accuracy scripts |
| Dashboard | Streamlit |
| Deployment | Render |
| CI | GitHub Actions  |
| Environment | Python 3.11, managed via Conda (`conda create -n gated python=3.11`) |

---

## 📁 Project Structure

```
it-triage-agent/
│
├── app/
│   ├── main.py                  # FastAPI entrypoint
│   ├── agents/
│   │   ├── classifier.py
│   │   ├── retriever.py
│   │   ├── resolver.py
│   │   └── escalation_judge.py
│   ├── graph.py                 # LangGraph state graph definition
│   ├── tools/                   # mock tool APIs
│   └── db/                      # models, pgvector setup
│
├── eval/
│   ├── eval_set.csv              # held-out tickets for testing
│   ├── run_eval.py
│   └── results/                  # eval output, confusion matrices
│
├── dashboard/
│   └── streamlit_app.py
│
├── notebooks/
│   └── data_exploration.ipynb
│
├── README.md
└── requirements.txt
```

- `sample_tickets_labeled.csv` — 20 hand-labeled tickets used to eval the Classifier Agent before the full dataset is wired in
- `test_classifier.py` — standalone script that runs `classify_ticket()` directly (no server needed) and reports category/priority accuracy
- `classifier_eval_results.csv` — output of the above, per-ticket predicted vs. expected

---

## ⚙️ Installation

```bash
git clone https://github.com/<Komal036>/ConfidenceGatedTriageAgent.git
cd ConfidenceGatedTriageAgent

# Create and activate a dedicated environment (Python 3.11)
conda create -n gated python=3.11
conda activate gated

pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Fill in GROQ_API_KEY (console.groq.com) and DATABASE_URL (Neon pooled connection string)

# Start the API — tables are created automatically on first run
uvicorn app.main:app --reload
```

Visit `http://localhost:8000/docs` for the interactive API tester.

---

## 🚀 Usage

### Submit a ticket

```bash
curl -X POST http://localhost:8000/submit-ticket \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Cannot connect to WiFi",
    "description": "My laptop keeps disconnecting every few minutes.",
    "product": "Dell XPS",
    "channel": "chat"
  }'
```

### Example response

```json
{
  "ticket_id": "T-1042",
  "category": "Network",
  "priority": "Medium",
  "action_taken": "escalated",
  "confidence": 0.42,
  "reasoning": "No close match found in knowledge base; ambiguous root cause.",
  "assigned_to": "human_queue"
}
```

---

## 🔍 Agent Design Deep-Dive

### Classifier Agent
- **Input:** ticket `subject` and `description` (free text)
- **Output:** structured JSON — `category` (one of 7 fixed options) and `priority` (Low/Medium/High/Critical)
- **Model:** Llama 3.3 70B via Groq, `temperature=0.1` (kept low deliberately — classification should be consistent, not creative)
- **Decision logic:** a single prompt with two distinct rubrics baked in — category guidance (distinguishing genuine issues from requests/questions, so "how do I upgrade my plan" isn't miscategorized as a Billing problem) and priority guidance (explicit criteria per level, e.g. "High = significant disruption with no workaround", rather than leaving urgency to the model's unguided judgment)
- **Failure handling:** if the LLM output can't be parsed as JSON, or returns a category/priority outside the fixed set, the agent falls back to `"General Inquiry"/"Medium"` and logs a warning rather than crashing the request — the same "fail safe, not silent" principle the Escalation Judge will later formalize with actual confidence scoring
- **Example:** "My laptop keeps disconnecting from WiFi" → `{"category": "Network", "priority": "Medium"}`

### Retriever Agent 
- Similarity threshold used: `0.55` (cosine similarity on `all-MiniLM-L6-v2` embeddings, 384-dim)
- What happens on no match: returns `None` rather than forcing a weak match. The
  Resolver Agent treats `None` as `"no_match"` and does not attempt to draft a
  reply — this is a deliberate handoff point for the Escalation Judge (Week 3)
  rather than a failure mode.


### Resolver Agent *(Week 2)*
- Tools available: `check_system_status`, `reset_password`, `lookup_account`, or `none`
- When it chooses to act vs draft-only: the LLM is given the matched knowledge
  base resolution as grounding and decides per-ticket whether a tool call would
  add information (e.g. confirming an outage) before drafting, or whether the
  reference resolution alone is enough to write a direct reply.


## 🚦 Escalation Logic


```
Escalation triggered when:
  confidence_score < THRESHOLD
  OR retrieved_match_similarity < MATCH_THRESHOLD
  OR category == "ambiguous"
```

Explain here **how you chose THRESHOLD** — via eval sweep, not guesswork. Show a small table of accuracy vs. false-escalation rate at different threshold values, and justify the one you picked.

---

## 📊 Evaluation Strategy

- Eval set: [N] tickets, mix of clear-cut and deliberately ambiguous cases
- Metrics tracked: resolution accuracy, false-escalation rate, false-confidence rate
- Method: [manual labeling / LLM-as-judge / hybrid — be explicit about which]

---

## 📈 Results
### Retriever Agent accuracy

Evaluated against the same 20 hand-labeled tickets used for the classifier eval
(`data/test_pipeline.py`), using a 25-entry hand-written knowledge base seeded
across all 7 categories.

| Metric | Result |
|---|---|
| Retrieval hit rate | 90.0% (18/20) |
| Category agreement (of hits) | 94.4% (17/18) |

**The 2 misses** were both genuine knowledge-base coverage gaps, not retrieval
failures: "screen flickering after update" has no matching entry (the closest
KB entry covers monitor *detection*, not flickering), and "feature request for
dark mode" has no matching entry because the KB's single General Inquiry entry
covers plan upgrades only. Both are addressed in Future Improvements (expand
KB with edge-case tickets).

**The 1 category mismatch** is more interesting: "wrong item billed" (Billing)
matched a General Inquiry entry about upgrading plans (similarity 0.619)
instead of a Billing entry that's a near-perfect fit — `"Charged for a plan
tier the user did not select"` — which exists in the KB but wasn't retrieved.
Both entries share surface vocabulary ("basic," "premium," "plan"), and the
embedding model appears to have weighted that lexical overlap over the
ticket's actual intent (an incorrect charge vs. a self-initiated upgrade
request). This is a legitimate retrieval error, not a coverage gap, and it
happened right at the threshold edge (0.619 vs. 0.55) — a case for future
threshold tuning or reranking rather than just adding more KB entries.
### Classifier Agent accuracy

| Metric | Iteration 1 (baseline prompt) | Iteration 2 (rubric-guided prompt) |
|---|---|---|
| Category accuracy | 75% (15/20)* | **100%** (20/20) |
| Priority accuracy | 60% (12/20)* | **75%** (15/20) |
| Both correct | 50% (10/20)* | **75%** (15/20) |

*\*Note: the baseline run's raw numbers included 3 tickets with a data-corruption bug (an unescaped comma in the CSV shifted columns); the real baseline, once corrected, was ~88% category / ~71% priority. Both iteration numbers above are on clean data.*

### What changed between iterations

The baseline prompt gave the model a list of valid categories/priorities with no criteria for choosing between them. Two failure patterns emerged:
1. **Category:** requests/questions with no actual problem (e.g. "how do I upgrade my plan") were being classified into whatever technical bucket seemed closest (Billing, Software) instead of `General Inquiry`.
2. **Priority:** the model consistently regressed toward `Medium` regardless of actual urgency — it avoided both `High` and `Low` even when the ticket clearly warranted them.

Adding explicit rubrics for both fixed category accuracy completely and improved priority substantially. The remaining priority misses in iteration 2 show a **new, smaller, opposite-direction bias**: the rubric's emphasis on "check for a full blocker before defaulting to Medium" made the model slightly *over*-escalate borderline cases (e.g. a WiFi reconnect issue with an easy workaround got called `High` instead of `Medium`). This is a legitimate, understood limitation — not random error — and is a good candidate for a future few-shot prompting pass (see Future Improvements).

---

## 💡 Key Design Decisions

### 1. Why LangGraph instead of CrewAI?

The core requirement driving this choice was auditability: every agent
decision needs to be inspectable after the fact via the `AgentDecision`
table, and eventually gated by an Escalation Judge that needs to see exactly
what each prior step decided and why. LangGraph's `StateGraph` makes the
pipeline's data flow explicit — a single `TriageState` TypedDict that each
node reads from and writes back to, with the sequencing defined as plain
edges (`classify → retrieve → resolve`). At any point, the full state is
inspectable, and adding a conditional branch (e.g. the Escalation Judge
routing "resolved but low-confidence" tickets differently than "no_match"
ones) is a matter of adding an edge condition, not restructuring how agents
communicate.

CrewAI's abstraction is built around roles and delegation — a crew of agents
that reason about how to divide work among themselves. That's a good fit
for open-ended tasks where the *sequence* of work isn't known in advance.
This project's pipeline is the opposite: the sequence is fixed and known
(classify, then retrieve, then resolve, then judge), and each step is a
plain Python function that's already unit-testable in isolation (see
`data/test_classifier.py`, `data/test_pipeline.py`). LangGraph fits that
shape more directly — it's an explicit graph over functions, not an
orchestration layer for agents deciding what to do next.

The trade-off: LangGraph gives up some of CrewAI's higher-level conveniences
(built-in agent-to-agent delegation, role prompting) in exchange for that
explicitness. For a pipeline where the whole point is a legible, auditable
decision path — not autonomous task delegation — that trade favors
LangGraph.

### 2. Why confidence-gating instead of full automation?
A fully autonomous agent that always resolves and always replies is easy to
build and easy to get badly wrong — a wrong password-reset trigger or a
wrong billing refund is worse than no action at all. The project's premise
is that *knowing when the pipeline doesn't know* is more valuable than
maximizing the percentage of tickets it touches.

This shows up in the design well before Week 3's formal Escalation Judge:
the Classifier falls back to a safe default rather than guessing on a parse
failure; the Retriever returns `None` instead of forcing a weak match below
`SIMILARITY_THRESHOLD`; the Resolver refuses to draft anything at all when
there's no retrieved match to ground it. Each agent already has a built-in
"I don't know" path — the Escalation Judge's job in Week 3 isn't to invent
this behavior, it's to formalize the existing per-agent confidence signals
(parse success, retrieval similarity, resolution grounding) into one
explicit gating decision, and to make that decision auditable via the
`AgentDecision` log rather than implicit in whichever agent happened to bail
out first.

The trade-off: this pipeline will escalate tickets a fully automated system
would have resolved correctly. That's the intended cost — the project treats
false escalation (a human reviews an easy ticket) as far cheaper than false
confidence (the system tells a user something wrong).

---
### 3. Why pgvector instead of a dedicated vector DB (Pinecone/Weaviate)?

The project already runs its relational data — `Ticket`, `AgentDecision` — on
Postgres via Neon. Adding pgvector meant the `Resolution` knowledge base could
live in the *same* database as one more table with a `Vector(384)` column,
rather than standing up a second system and syncing state between them.

For a knowledge base this size (25 entries now, realistically hundreds even
after Future Improvements' "expand with synthetic edge cases"), a dedicated
vector DB's main selling points — approximate nearest-neighbor indexes for
millions of vectors, horizontal sharding — don't apply. pgvector's exact
cosine-distance search (`<=>` operator) over a few hundred rows runs in
single-digit milliseconds with no index at all; an IVFFlat or HNSW index
would be premature optimization here.

The trade-off is real, though: this only holds because the KB is small and
single-tenant. If the knowledge base grew into the tens of thousands of
entries, or needed to scale independently of the transactional data, a
dedicated vector store's ANN indexing would start to matter, and that
coupling to Postgres would become the thing to undo.

---
### 4. Why Groq/Llama instead of GPT-4?

Two practical reasons.

First, latency: Groq's inference is fast enough that running three
sequential LLM calls per ticket (Classifier → Resolver, with the Retriever's
embedding step in between) doesn't compound into a noticeably slow request —
this matters more for a chained pipeline than it would for a single
standalone call.

Second, cost: at this project's scale (hand-labeled 20-ticket eval sets,
25-entry KB, no production traffic), Groq's free tier and Llama 3.3 70B's
pricing make iteration cheap. The classifier's two-iteration prompt rework
(baseline → rubric-guided) and the resolver's prompt tuning both involved
re-running the full eval set repeatedly — a cost-sensitive workflow benefits
from a cheaper model here.

The trade-off is real: Llama 3.3 70B is a smaller, less capable model than
GPT-4, and it shows in places — the documented priority over-escalation bias
in the classifier is partly a prompting gap, but a larger model might have
generalized better from the same rubric with less explicit spelling-out.
For a project demonstrating agent *architecture* (multi-agent orchestration,
confidence gating, tool use) rather than pushing single-model classification
accuracy to its ceiling, that trade-off favors Groq/Llama.

---

## 🔮 Future Improvements

### Short term
- [ ] Expand knowledge base with synthetic edge-case tickets
- [ ] Add per-category confidence thresholds instead of one global threshold

### Medium term
- [ ] Replace mock tools with a real ticketing system integration (Zendesk/Freshdesk sandbox API)
- [ ] Add a feedback loop where human corrections retrain the Retriever's knowledge base

### Long term
- [ ] Multi-turn ticket handling (follow-up questions before resolving)
- [ ] A/B test against a single-LLM baseline to quantify the value of the multi-agent structure

---

## 📐 Metrics Explained

### Why not just accuracy?

Plain accuracy treats a false-resolution and a false-escalation as equally bad — they aren't. A false resolution (wrongly closing a ticket) damages user trust and can hide a real problem. A false escalation just costs a bit of human time. This project tracks them **separately** so the tradeoff is visible and tunable, not hidden inside one aggregate number.

---

## 📚 References

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/)
- Dataset: [Customer Support Ticket Dataset — Kaggle](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset)

---

**Built with FastAPI · LangGraph · Groq · pgvector**
