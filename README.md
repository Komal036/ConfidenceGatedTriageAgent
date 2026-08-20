# 🎧 Autonomous IT Support Triage & Resolution Agent

### Multi-Agent System for Confidence-Gated Ticket Classification, Retrieval, and Escalation

## 🔗 Live Demo

- **App:** https://confidence-gated-triage-agent.vercel.app
- **API docs:** https://confidencegatedtriageagent.onrender.com/docs

_Note: the backend runs on a free tier that sleeps after inactivity — the first request may take up to a minute to wake up._

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?style=flat-square&logo=fastapi)
![LangGraph](https://img.shields.io/badge/LangGraph-orchestration-purple?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-GPT--OSS_120B-orange?style=flat-square)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-blue?style=flat-square&logo=postgresql)
![Dataset](https://img.shields.io/badge/Dataset-8.4k_tickets-yellow?style=flat-square)
[![CI](https://github.com/Komal036/ConfidenceGatedTriageAgent/actions/workflows/ci.yml/badge.svg)](https://github.com/Komal036/ConfidenceGatedTriageAgent/actions/workflows/ci.yml)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Limitations](#-limitations)
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
- [Key Learnings](#-key-learnings)
- [Future Improvements](#-future-improvements)
- [Metrics Explained](#-metrics-explained)
- [References](#-references)

---

## 🎯 Overview

This project is a multi-agent system that automates first-line IT support ticket handling. Given an incoming ticket (subject, description, product/category context), the system classifies it, checks for a known resolution, attempts an autonomous fix via tool calls, and critically **knows when not to act**, escalating to a human agent when its own confidence is low.

**Result (20-ticket held-out eval): 70% category accuracy, 50% priority accuracy, 45% escalation decision accuracy, 100% false-escalation rate, 0% false-confidence rate.** The system currently errs heavily toward caution given a 50-entry knowledge base's ~26% real-world match rate — see Results for the full breakdown of why that's a deliberate, honest tradeoff rather than a failure, and Future Improvements for the concrete next steps (a stronger reranker, further KB expansion) that would move these numbers.

_Note: this project's LLM originally ran on Llama 3.3 70B and was migrated to GPT-OSS 120B mid-project after Groq deprecated the former (see Key Design Decisions #5). Some results below were measured on one model, some on the other — each results section is labeled with which model produced it._

---

## ⚠️ Limitations

Stated bluntly, up front, rather than only inferable from the Results section below:

- **This is a rigorously evaluated prototype, not a production system.** There is no auth, no rate limiting, and no observability/logging stack beyond the `AgentDecision` audit table. Don't read "confidence-gated" or "evaluated" as "production-ready" — they're not the same claim.
- **The knowledge base is small and hand-written (50 entries).** It matches roughly a quarter of real-world-phrased tickets from the Kaggle eval set. This is the actual ceiling on end-to-end accuracy right now — see Results and Future Improvements for why (an embedding-model limitation, not a threshold-tuning or query-construction bug).
- **Ground truth labels carry a circularity risk that isn't fully resolved.** Eval labels were LLM-drafted, then manually reviewed and corrected by the same person who wrote the classification rubric the classifier itself is prompted with (see Evaluation Strategy). That means classifier and ground truth share an author and a mental model of what "correct" looks like — an independent labeler, or an inter-rater agreement check against a second reviewer, would give a cleaner signal than self-review alone can. Treat the reported accuracy numbers as directionally honest but not fully independent.
- **This reads as agent/backend engineering work, not full-stack.** The frontend (Next.js) is a real, hand-built interface, but the bulk of the engineering investment — and the code Komal can speak to in depth — is the backend pipeline, retrieval, and evaluation harness.
- **Unit test coverage exists but has a real edge:** `tests/` covers FastAPI routes, DB schema, and the Escalation Judge's decision logic, all offline via mocking. It does not include a live-database integration test in CI (no pgvector-enabled Postgres service is wired into the GitHub Actions job) or assertions about LLM output _quality_ — that's what the eval scripts in `data/` are for, and their numbers are the ones to trust for "does this actually work," not the pytest suite.

---

## 📋 Problem Statement

| Property  | Details                                                            |
| --------- | ------------------------------------------------------------------ |
| Task      | Multi-class classification + retrieval + gated autonomous action   |
| Input     | Ticket subject, description, product/category metadata             |
| Output    | Category, priority, resolution action OR escalation with reasoning |
| Metric    | Resolution accuracy, false-escalation rate, false-confidence rate  |
| Dataset   | [8,469 tickets — Kaggle Customer Support Ticket Dataset]           |
| Challenge | Balancing autonomy against the cost of being confidently wrong     |

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

| Category            | Technology                                                                                                |
| ------------------- | --------------------------------------------------------------------------------------------------------- |
| Backend             | FastAPI 0.115, Uvicorn, Pydantic v2, SQLAlchemy 2.0                                                       |
| Database            | PostgreSQL (Neon, pooled connection)                                                                      |
| Vector store        | pgvector                                                                                                  |
| Agent orchestration | LangGraph, LangChain                                                                                      |
| LLM inference       | Groq API — GPT-OSS 120B (`groq==1.6.0`), migrated from Llama 3.3 70B after Groq deprecated it mid-project |
| Embeddings          | sentence-transformers (all-MiniLM-L6-v2)                                                                  |
| Evaluation          | pandas, custom accuracy scripts                                                                           |
| Dashboard           | Streamlit                                                                                                 |
| Deployment          | Render                                                                                                    |
| CI                  | GitHub Actions                                                                                            |
| Environment         | Python 3.11, managed via Conda (`conda create -n gated python=3.11`)                                      |

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
├── tests/                       # pytest unit tests (routes, DB schema, agents)
│   ├── test_routes.py
│   ├── test_escalation_judge.py
│   ├── test_classifier_agent.py
│   └── test_db_models.py
│
├── .github/workflows/ci.yml     # runs the pytest suite on push/PR
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

- `tests/` — pytest unit tests, distinct in purpose from the eval scripts above: these check code correctness (route validation, response shape, escalation logic, DB schema), not model accuracy. The Groq client and DB session are mocked so the suite runs offline in CI with no API key or live database. Run locally with `pytest tests/ -v`.
- `sample_tickets_labeled.csv` — 20 hand-labeled tickets used to eval the Classifier Agent before the full dataset is wired in
- `test_classifier.py` — standalone script that runs `classify_ticket()` directly (no server needed) and reports category/priority accuracy
- `classifier_eval_results.csv` — output of the above, per-ticket predicted vs. expected
- `diagnose_retrieval_scores.py` — prints the raw best-match similarity for every tune-set ticket, bypassing the Retriever's threshold cutoff, for debugging coverage gaps
- `seed_knowledge_base.py` / `seed_knowledge_base_data.py` — clears and re-seeds the pgvector knowledge base table from the hand-written entry list

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
  "id": "3f9a1c2e-8b4d-4e91-9c3a-1a2b3c4d5e6f",
  "subject": "Cannot connect to WiFi",
  "description": "My laptop keeps disconnecting every few minutes.",
  "category": "Network",
  "priority": "Medium",
  "status": "resolved",
  "matched_issue": "WiFi keeps disconnecting intermittently",
  "match_similarity": 0.746,
  "draft_resolution": "Try restarting your router, forgetting and rejoining the network, and updating your WiFi adapter drivers.",
  "tool_called": null,
  "escalated": false,
  "escalation_reason": "Match similarity 0.746 meets the confidence bar (0.6) and priority ('Medium') is non-critical."
}
```

---

## 🔍 Agent Design Deep-Dive

### Classifier Agent

- **Input:** ticket `subject` and `description` (free text)
- **Output:** structured JSON — `category` (one of 7 fixed options) and `priority` (Low/Medium/High/Critical)
- **Model:** `openai/gpt-oss-120b` via Groq (migrated from Llama 3.3 70B — see Key Design Decisions), `temperature=0.1` (kept low deliberately — classification should be consistent, not creative), `reasoning_effort="low"` (see Key Design Decisions #5 — GPT-OSS's default reasoning mode was silently consuming the entire token budget before producing output)
- **Decision logic:** a single prompt with two distinct rubrics baked in — category guidance (distinguishing genuine issues from requests/questions, so "how do I upgrade my plan" isn't miscategorized as a Billing problem) and priority guidance (explicit criteria per level, e.g. "High = significant disruption with no workaround", rather than leaving urgency to the model's unguided judgment)
- **Failure handling:** if the LLM output can't be parsed as JSON, or returns a category/priority outside the fixed set, the agent falls back to `"General Inquiry"/"Medium"` and logs a warning rather than crashing the request — the same "fail safe, not silent" principle the Escalation Judge later formalizes with actual confidence scoring
- **Example:** "My laptop keeps disconnecting from WiFi" → `{"category": "Network", "priority": "Medium"}`

### Retriever Agent

- Similarity threshold used: `0.55` (cosine similarity on `all-MiniLM-L6-v2` embeddings, 384-dim)
- What happens on no match: returns `None` rather than forcing a weak match. The
  Resolver Agent treats `None` as `"no_match"` and does not attempt to draft a
  reply — this is a deliberate handoff point for the Escalation Judge
  rather than a failure mode.

### Resolver Agent

- Tools available: `check_system_status`, `reset_password`, `lookup_account`, or `none`
- When it chooses to act vs draft-only: the LLM is given the matched knowledge
  base resolution as grounding and decides per-ticket whether a tool call would
  add information (e.g. confirming an outage) before drafting, or whether the
  reference resolution alone is enough to write a direct reply.

---

## 🚦 Escalation Logic

Escalates to a human when ANY of:

```
resolution_status != "resolved"    # no confident KB match at all
OR match_similarity < 0.60          # match found, but too weak to trust
OR priority == "Critical"           # always human-reviewed regardless of match quality
```

`0.60` was chosen via a threshold sweep against 50 hand-labeled real
Kaggle tickets (`data/sweep_escalation_threshold.py`), run through the
actual Classifier + Retriever pipeline (not the hand labels):

| Threshold | Accuracy  | False-Escalation Rate | False-Confidence Rate |
| --------- | --------- | --------------------- | --------------------- |
| 0.55      | 46.0%     | 76.7%                 | 20.0%                 |
| **0.60**  | **48.0%** | 83.3%                 | **5.0%**              |
| 0.65      | 44.0%     | 93.3%                 | 0.0%                  |
| 0.70–0.90 | 40.0%     | 100.0%                | 0.0%                  |

0.65 has a lower false-confidence rate on paper, but it sits above every
real match similarity observed in the eval set (max 0.678) — at that
point the Judge isn't making a tuned decision, it's just escalating
nearly everything, which trivially drives the "costly" error to zero at
the expense of being a functioning system at all. 0.60 is the actual
accuracy peak, trading one additional false-confidence case for a
meaningfully better false-escalation rate.

---

## 📊 Evaluation Strategy

- **Eval sets:** 70 real tickets sampled from the Kaggle Customer Support Ticket Dataset (`data/pull_new_tickets.py`), split into `eval_tune.csv` (50 tickets, used to sweep the Escalation Judge's threshold and diagnose KB coverage) and `eval_holdout.csv` (20 tickets, never touched during tuning, used only for final reported numbers) — this split exists specifically so reported accuracy isn't inflated by having been part of the same search that picked the threshold.
- **Labeling method:** hybrid. Labels were LLM-drafted (category, priority, escalate) then manually reviewed and corrected row by row against the project's own rubric (see Agent Design Deep-Dive) before being treated as ground truth. _This introduces a circularity risk worth naming directly: the same rubric shapes both the classifier's prompt and the reviewer's corrections, so accuracy numbers reflect internal consistency with that rubric more than independent ground truth — see Limitations._
- **Data cleaning:** the raw Kaggle descriptions required a cleaning pass (`data/clean_eval_descriptions.py`) before use — see Key Learnings for why, and Results for the retrieval-generalization gap this process uncovered.
- **Metrics tracked:** category accuracy, priority accuracy, escalation decision accuracy, false-escalation rate, false-confidence rate — tracked separately rather than as one blended accuracy number, since a false resolution and a false escalation are not equally costly (see Metrics Explained).

---

## 📈 Results

### Classifier Agent accuracy

_Note: the results below were measured on the original `llama-3.3-70b-versatile`
model, before Groq's deprecation forced a migration to `openai/gpt-oss-120b`
(see Key Design Decisions #5). They are not directly comparable to the Final
Holdout Evaluation below, which runs on the current model._

| Metric            | Iteration 1 (baseline prompt) | Iteration 2 (rubric-guided prompt) |
| ----------------- | ----------------------------- | ---------------------------------- |
| Category accuracy | 75% (15/20)\*                 | **100%** (20/20)                   |
| Priority accuracy | 60% (12/20)\*                 | **75%** (15/20)                    |
| Both correct      | 50% (10/20)\*                 | **75%** (15/20)                    |

_\*Note: the baseline run's raw numbers included 3 tickets with a data-corruption bug (an unescaped comma in the CSV shifted columns); the real baseline, once corrected, was ~88% category / ~71% priority. Both iteration numbers above are on clean data._

**What changed between iterations:** the baseline prompt gave the model a list of valid categories/priorities with no criteria for choosing between them. Two failure patterns emerged:

1. **Category:** requests/questions with no actual problem (e.g. "how do I upgrade my plan") were being classified into whatever technical bucket seemed closest (Billing, Software) instead of `General Inquiry`.
2. **Priority:** the model consistently regressed toward `Medium` regardless of actual urgency — it avoided both `High` and `Low` even when the ticket clearly warranted them.

Adding explicit rubrics for both fixed category accuracy completely and improved priority substantially. The remaining priority misses in iteration 2 show a **new, smaller, opposite-direction bias**: the rubric's emphasis on "check for a full blocker before defaulting to Medium" made the model slightly _over_-escalate borderline cases (e.g. a WiFi reconnect issue with an easy workaround got called `High` instead of `Medium`). This is a legitimate, understood limitation — not random error — and is a good candidate for a future few-shot prompting pass (see Future Improvements).

### Retriever Agent accuracy (hand-written eval set)

Evaluated against the original 20 hand-labeled hand-written tickets
(`data/test_pipeline.py`), using the initial 25-entry hand-written knowledge
base seeded across all 7 categories.

| Metric                       | Result        |
| ---------------------------- | ------------- |
| Retrieval hit rate           | 90.0% (18/20) |
| Category agreement (of hits) | 94.4% (17/18) |

**The 2 misses** were both genuine knowledge-base coverage gaps, not retrieval
failures: "screen flickering after update" has no matching entry (the closest
KB entry covers monitor _detection_, not flickering), and "feature request for
dark mode" has no matching entry because the KB's single General Inquiry entry
covers plan upgrades only.

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

### KB coverage on real-world tickets (two expansion rounds)

Running the pipeline against 50 real (not hand-written) Kaggle-sourced
tickets exposed a generalization gap: the original 25-entry KB, hand-written
to match the phrasing of eval tickets, matched **0% of real tickets** above
the Retriever's 0.55 threshold.

**Round 1** expanded the KB to 45 entries with more naturally-phrased issue
summaries (targeting patterns actually observed in the real ticket sample —
firmware-triggered faults, factory-reset-didn't-help, peripheral/charging
issues, vague security concerns — written independently, not copied from
the eval tickets themselves), raising the real-ticket match rate to **24%
(12/50)**.

**Round 2** analyzed the 38 remaining misses directly against their real
descriptions (not just subjects, since Kaggle's synthetic subject labels are
frequently mismatched to the actual complaint — e.g. a subject of "Delivery
problem" on a ticket that's actually about data loss). This surfaced further
concrete gaps (declining battery life vs. sudden drain, first-setup WiFi
failures, vague/unlabeled error messages, "can't find a feature" navigation
requests, widespread multi-user bugs, general security worry). Five targeted
entries were added, bringing the KB to 50 entries and raising the match rate
to **26% (13/50)** — a real but small gain relative to Round 1.

**Diagnosing the diminishing returns:** a controlled experiment
(`data/diagnose_row3_anomaly.py`) isolated why. One remaining miss — a
ticket asking "I'm unable to find the option to perform the desired action,
could you guide me through the steps?" — scored only **0.216** against a KB
entry that is, in meaning, a near-exact paraphrase: _"user can't locate a
specific feature, setting, or option and needs step-by-step navigation
guidance."_ A sanity check against clearly unrelated KB entries (WiFi,
battery, billing) confirmed the embedding model's _ranking_ is correct
(the real match scores far higher than unrelated ones), but its **absolute
similarity score for this kind of paraphrase pair is intrinsically low** —
`all-MiniLM-L6-v2` leans more on lexical/structural overlap than deep
paraphrase understanding, since the two sentences share almost no surface
vocabulary despite matching in meaning.

**This is the real ceiling on further KB-only improvements**: no amount of
additional hand-written entries fixes a case where the correct entry
already exists but the embedding model itself under-scores the semantic
match. Closing this gap further would need either a stronger embedding
model or a reranking step (e.g. a cross-encoder) rather than more KB
content — see Future Improvements.

### Final holdout evaluation (current model: GPT-OSS 120B, 50-entry KB)

Evaluated end-to-end (Classifier → Retriever → Resolver → Escalation Judge)
against 20 real Kaggle support tickets held out from threshold tuning.

| Metric                       | Value         |
| ---------------------------- | ------------- |
| Classifier category accuracy | 70.0% (14/20) |
| Classifier priority accuracy | 50.0% (10/20) |
| Escalation decision accuracy | 45.0% (9/20)  |
| False-escalation rate        | 100% (11/11)  |
| False-confidence rate        | 0% (0/9)      |

**Zero false-confidence is the load-bearing number here.** The system never
auto-resolved a ticket it shouldn't have — every escalation "failure" was an
_over-cautious_ one (escalating a ticket a human reviewer might have let
through), never an under-cautious one. For a confidence-gated system, this
is the correct failure direction: it's far safer to escalate unnecessarily
than to auto-resolve incorrectly.

**On escalation:** the 100% false-escalation rate looks alarming in
isolation, but the underlying cause is consistent and traceable, not a
broken system. Of the 20 holdout tickets, only 4 retrieved any KB match at
all (a 20% hit rate — close to the ~26% match rate observed on the 50-ticket
tune set after two rounds of KB expansion, within expected small-sample
variance). The other 16 tickets found no match at all, which alone forces
escalation regardless of priority or threshold tuning. Two rounds of KB
expansion (0%→24%→26%) and a controlled embedding-model diagnostic (see
above) point to this as a genuine, largely embedding-model-limited ceiling
rather than an unexamined gap — the threshold itself isn't the limiting
factor here.

**On category accuracy (70.0%, 14/20):** up from 60.0% in the previous run
on the same holdout set. This shift is not attributable to the KB expansion
(the Classifier doesn't use the knowledge base at all) and is more likely
run-to-run variance from the LLM call, since `temperature=0.1` doesn't
guarantee full determinism. Of the 6 remaining misses, no single dominant
default pattern was observed this round (unlike the earlier "Hardware"
default-under-uncertainty signal seen in a prior run) — not yet enough
evidence to confirm or rule out a systematic bias.

**On priority accuracy (50.0%, 10/20):** predictions continue to skew
toward High/Critical on tickets expected to be Medium. This did not affect
any escalation decisions in this run, since none of the 20 tickets reached
the priority-based override path (all escalations were driven by match
confidence, not priority) — but it's a real classifier weakness worth
addressing independently. See Future Improvements.

---

## 💡 Key Design Decisions

### 1. Why LangGraph instead of CrewAI?

The core requirement driving this choice was auditability: every agent
decision needs to be inspectable after the fact via the `AgentDecision`
table, and gated by an Escalation Judge that needs to see exactly what each
prior step decided and why. LangGraph's `StateGraph` makes the pipeline's
data flow explicit — a single `TriageState` TypedDict that each node reads
from and writes back to, with the sequencing defined as plain edges
(`classify → retrieve → resolve → judge`). At any point, the full state is
inspectable, and adding a conditional branch is a matter of adding an edge
condition, not restructuring how agents communicate.

CrewAI's abstraction is built around roles and delegation — a crew of agents
that reason about how to divide work among themselves. That's a good fit
for open-ended tasks where the _sequence_ of work isn't known in advance.
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
is that _knowing when the pipeline doesn't know_ is more valuable than
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
confidence (the system tells a user something wrong). The holdout eval is
the clearest demonstration of this in practice: 0% false confidence, at the
cost of a high false-escalation rate on a KB that hasn't reached full
real-world coverage yet.

### 3. Why pgvector instead of a dedicated vector DB (Pinecone/Weaviate)?

The project already runs its relational data — `Ticket`, `AgentDecision` — on
Postgres via Neon. Adding pgvector meant the `Resolution` knowledge base could
live in the _same_ database as one more table with a `Vector(384)` column,
rather than standing up a second system and syncing state between them.

For a knowledge base this size (50 entries as of Week 4, realistically a few
hundred even after further expansion), a dedicated vector DB's main selling
points — approximate nearest-neighbor indexes for millions of vectors,
horizontal sharding — don't apply. pgvector's exact cosine-distance search
(`<=>` operator) over a few hundred rows runs in single-digit milliseconds
with no index at all; an IVFFlat or HNSW index would be premature
optimization here.

The trade-off is real, though: this only holds because the KB is small and
single-tenant. If the knowledge base grew into the tens of thousands of
entries, or needed to scale independently of the transactional data, a
dedicated vector store's ANN indexing would start to matter, and that
coupling to Postgres would become the thing to undo.

### 4. Why Groq/Llama instead of GPT-4?

Two practical reasons.

First, latency: Groq's inference is fast enough that running multiple
sequential LLM calls per ticket (Classifier → Resolver, with the Retriever's
embedding step and the Escalation Judge's rule check in between) doesn't
compound into a noticeably slow request — this matters more for a chained
pipeline than it would for a single standalone call.

Second, cost: at this project's scale (hand-labeled eval sets, a 50-entry
KB, no production traffic), Groq's free tier and open-weight model pricing
make iteration cheap. The classifier's two-iteration prompt rework
(baseline → rubric-guided), the resolver's prompt tuning, and the threshold
sweep (50 tickets × Classifier + Retriever calls, run more than once across
KB expansion rounds) all involved re-running eval sets repeatedly — a
cost-sensitive workflow benefits from a cheaper model here.

The trade-off is real: smaller open-weight models are less capable than
GPT-4, and it shows in places — the classifier's priority over-escalation
bias (observed on both Llama 3.3 70B and, independently, on GPT-OSS 120B
after migration — see Key Design Decisions #5) suggests this is a rubric
tuning gap rather than a single-model quirk. A larger, more heavily-aligned
model might generalize past it with less explicit rubric spelling-out. For
a project demonstrating agent _architecture_ (multi-agent orchestration,
confidence gating, tool use) rather than pushing single-model classification
accuracy to its ceiling, that trade-off favors Groq's open-weight models
over GPT-4.

Note: this project's LLM choice was originally Llama 3.3 70B; it now runs
on GPT-OSS 120B after Groq deprecated the former mid-project (see Key
Design Decisions #5). The reasoning above regarding Groq vs. GPT-4 still
applies to the current model.

### 5. Handling a mid-project model deprecation (Llama 3.3 70B → GPT-OSS 120B)

Partway through development, Groq deprecated `llama-3.3-70b-versatile` (the
model this project was originally built on) for free/developer-tier usage,
replacing it with `openai/gpt-oss-120b`. The migration wasn't a one-line
swap — it surfaced a real debugging problem worth documenting.

**First attempt (incomplete):** updating the model string and adding
`response_format={"type": "json_object"}` plus a higher `max_tokens` fixed
the Resolver, but the Classifier still failed to parse LLM output on
**100% of tickets** — every call returned an empty response.

**Root cause:** `gpt-oss-120b` is a _reasoning_ model — by default
(`reasoning_effort="medium"`) it spends part of its token budget on hidden
chain-of-thought before producing the actual JSON answer. With the
Classifier's tighter `max_tokens` budget, that reasoning consumed the
entire allowance before any output was generated, leaving nothing to parse.

**Actual fix:** setting `reasoning_effort="low"` on both the Classifier and
Resolver, alongside `response_format={"type": "json_object"}` and a modest
`max_tokens` increase, resolved it completely — 0 parse failures across
subsequent eval runs.

This is a real constraint of building on third-party LLM APIs: model
availability isn't guaranteed to stay static, and newer models in the same
family can have materially different default behavior (like hidden
reasoning tokens) that silently breaks assumptions baked into prompts and
token budgets tuned for a prior model.

---

## 🐛 Key Learnings

### A schema/frontend mismatch that made the Confidence Gate lie

The Usage section above documents `escalated` and `escalation_reason` as
part of `/submit-ticket`'s response — and the frontend's `TicketResult`
type in `lib/api.ts` was written against that documented contract. But
`TicketResponse` in `app/schemas.py` never actually defined those two
fields, so the real API response silently omitted them despite the docs
and the frontend type both assuming they were there.

At runtime, `result.escalated` was always `undefined`. The Confidence
Gate's `open` state was computed as `!result.escalated` — and
`!undefined` evaluates to `true` in JavaScript. The gate showed
**"OPEN — AUTO-RESOLVED" for every single ticket**, regardless of what
the Escalation Judge actually decided, including tickets the backend had
genuinely flagged for human escalation. The reasoning box was blank for
the same underlying reason — `escalation_reason` was undefined too.

**How it was caught:** not by the test suite (this predates the tests in
`tests/`, and it's exactly the kind of gap a schema-level test alone
wouldn't have caught, since Pydantic will happily validate a response
that's simply missing optional context the frontend expected). It
surfaced by deliberately cross-checking three independent sources against
each other for the same ticket: the `AgentDecision` audit rows in the
database (which correctly showed `"No confident match found"`), the raw
`/submit-ticket` response body inspected directly in the browser's
Network tab, and the UI's rendered state. The audit trail said one thing,
the UI said another — that mismatch is what pointed at the frontend
rather than the escalation logic itself, which turned out to be correct
all along.

**The fix, two-sided:**

1. `app/schemas.py` — added `escalated: bool` and `escalation_reason: str`
   to `TicketResponse`, matching what was already documented.
2. `app/main.py` — `submit_ticket()` now actually reads
   `result["escalate"]` / `result["escalation_reason"]` (already computed
   by `judge_node` in the LangGraph pipeline, just never returned to the
   client) and populates them. This also surfaced a second gap: the
   Escalation Judge's decision was never written to the `AgentDecision`
   audit table at all — classifier, retriever, and resolver each got a
   row, the Judge didn't — which is now fixed alongside it.

**Regression coverage:** `tests/test_routes.py`'s
`TestEscalationFieldsInResponse` class pins this at the API-response
level specifically — not just that `judge_escalation()` computes the
right answer internally (that's `test_escalation_judge.py`'s job), but
that `submit_ticket()` actually puts it on the response. Confirmed by
running these tests against the pre-fix code: they fail with
`KeyError: 'escalated'`, and pass against the fix.

**The general lesson:** a documented API contract and a frontend type
both agreeing on a field's existence isn't the same as the backend
actually sending it. Nothing in the type system or the docs enforced
that connection — `res.json()` in `lib/api.ts` was an untyped cast, so
TypeScript had no way to catch the mismatch at compile time, and it took
a live, cross-source check (DB vs. wire response vs. UI) to catch a bug
that made a "confidence-gated" system look confident about everything.

---

## 🔮 Future Improvements

### Short term

- [ ] Add a reranking step (e.g. a cross-encoder) for borderline retrieval
      cases — the row-3 diagnostic (see Results) showed the embedding model
      correctly ranks the right KB entry highest but under-scores its
      absolute similarity for low-lexical-overlap paraphrases; a reranker
      operating on the top-k candidates could recover these without
      further KB expansion
- [ ] Add per-category confidence thresholds instead of one global threshold
- [ ] Expand holdout eval set beyond 20 tickets — current sample is too
      small to reliably observe the auto-resolve path given a ~20-26% KB
      match rate
- [ ] Investigate the classifier's priority over-escalation bias with a
      few-shot prompting pass, since rubric guidance alone hasn't fully
      resolved it across two model generations

### Medium term

- [ ] Replace mock tools with a real ticketing system integration (Zendesk/Freshdesk sandbox API)
- [ ] Add a feedback loop where human corrections retrain the Retriever's knowledge base
- [ ] Evaluate a larger/stronger sentence embedding model as an alternative to `all-MiniLM-L6-v2`, trading some latency for better paraphrase sensitivity

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
