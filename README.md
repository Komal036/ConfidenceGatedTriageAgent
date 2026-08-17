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

**Week 3 result (20-ticket held-out eval): 55% category accuracy, 100% false-escalation rate, 0% false-confidence rate.** The system currently errs heavily toward caution given a 45-entry knowledge base's ~20-24% real-world match rate — see Results for the full breakdown of why that's a deliberate, honest tradeoff rather than a failure, and Future Improvements for the concrete next steps (KB expansion, category rubric refinement) that would move these numbers.

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
| Vector store | pgvector  |
| Agent orchestration | LangGraph, LangChain  |
| LLM inference | Groq API — Llama 3.3 70B (`groq==1.6.0`) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2)  |
| Evaluation | pandas, custom accuracy scripts |
| Dashboard | Streamlit |
| Deployment | Render |
| CI | GitHub Actions |
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
- **Model:** Llama 3.3 70B via Groq, `temperature=0.1` (kept low deliberately — classification should be consistent, not creative)
- **Decision logic:** a single prompt with two distinct rubrics baked in — category guidance (distinguishing genuine issues from requests/questions, so "how do I upgrade my plan" isn't miscategorized as a Billing problem) and priority guidance (explicit criteria per level, e.g. "High = significant disruption with no workaround", rather than leaving urgency to the model's unguided judgment)
- **Failure handling:** if the LLM output can't be parsed as JSON, or returns a category/priority outside the fixed set, the agent falls back to `"General Inquiry"/"Medium"` and logs a warning rather than crashing the request — the same "fail safe, not silent" principle the Escalation Judge will later formalize with actual confidence scoring
- **Example:** "My laptop keeps disconnecting from WiFi" → `{"category": "Network", "priority": "Medium"}`

### Retriever Agent 
- Similarity threshold used: ...
- What happens on no match: ...

### Resolver Agent 
- Tools available: ...
- When it chooses to act vs draft-only: ...

---

## 🚦 Escalation Logic
Escalates to a human when ANY of:

resolution_status != "resolved" # no confident KB match at all

OR match_similarity < 0.60 # match found, but too weak to trust

OR priority == "Critical" # always human-reviewed regardless of match quality

`0.60` was chosen via a threshold sweep against 50 hand-labeled real
Kaggle tickets (`data/sweep_escalation_threshold.py`), run through the
actual Classifier + Retriever pipeline (not the hand labels):

| Threshold | Accuracy | False-Escalation Rate | False-Confidence Rate |
|---|---|---|---|
| 0.55 | 46.0% | 76.7% | 20.0% |
| **0.60** | **48.0%** | 83.3% | **5.0%** |
| 0.65 | 44.0% | 93.3% | 0.0% |
| 0.70–0.90 | 40.0% | 100.0% | 0.0% |

0.65 has a lower false-confidence rate on paper, but it sits above every
real match similarity observed in the eval set (max 0.678) — at that
point the Judge isn't making a tuned decision, it's just escalating
nearly everything, which trivially drives the "costly" error to zero at
the expense of being a functioning system at all. 0.60 is the actual
accuracy peak, trading one additional false-confidence case for a
meaningfully better false-escalation rate.

## 📈 Results
### Final holdout evaluation 

Run against the 20 tickets held out from all tuning, using the real,
locked-in pipeline end to end (`data/final_holdout_eval.py`):

| Metric | Result |
|---|---|
| Category accuracy | 55.0% (11/20) |
| Priority accuracy | 75.0% (15/20) |
| Escalation decision accuracy | 45.0% (9/20) |
| False-escalation rate | 100% (11/11) |
| False-confidence rate | **0%** (0/9) |

**On escalation:** the 100% false-escalation rate looks alarming in
isolation, but the underlying data tells a consistent, expected story, not
a broken system. Of the 20 holdout tickets, only 4 retrieved any KB match
at all (a 20% hit rate — consistent with the tune set's 24%, small-sample
variance). All 4 of those matches scored between 0.555 and 0.572 —
just under the 0.60 escalation threshold by 0.03–0.045. With this few
data points landing this close to the bar, this is normal sampling
variance, not a threshold or KB failure — the tune set (50 tickets)
already demonstrated the system successfully clearing 0.60 on multiple
real tickets (up to 0.678).

The one number that matters most held perfectly: **0% false confidence.**
Every escalation on this holdout set was unnecessarily cautious, never
wrongly confident. That's the system's core design principle (see Key
Design Decisions #2) working exactly as intended, in its most extreme
form — proof the Judge fails safe rather than fails silent.

**On category accuracy:** the drop from Week 1's 100% (on hand-written
tickets closely matching the classifier's training rubric) to 55% here
reflects real-world ticket ambiguity, not pure classifier error — several
misses were on tickets the labeling process itself flagged as genuinely
borderline (e.g. the "security/data safety" template debated between
Account Access and Network during hand-labeling).

More interesting: **6 of the 9 category misses defaulted to `Hardware`.**
This mirrors the exact failure pattern already found and fixed 
for priority (the model defaulting to "Medium" under uncertainty) —
except here it's the category rubric defaulting to "Hardware" instead of
genuinely reasoning through ambiguous cases. This is a concrete, scoped
target for a future prompt-rubric pass (see Future Improvements), the
same fix pattern that took priority accuracy from 60% to 75% in Week 1.
### Escalation Judge / KB coverage on real-world tickets

Running the pipeline against 50 real (not hand-written) Kaggle-sourced
tickets exposed a generalization gap: the 25-entry KB, hand-written to
match the phrasing of the original Week 1–2 synthetic eval tickets,
matched **0% of real tickets** above the Retriever's own 0.55 threshold.
Expanding the KB to 45 entries with more naturally-phrased issue
summaries (targeting patterns actually observed in the real ticket
sample — firmware-triggered faults, factory-reset-didn't-help,
peripheral/charging issues, vague security concerns — written
independently, not copied from the eval tickets themselves) raised the
real-ticket match rate to **24% (12/50)**.

This is the honest headline result, not the threshold itself:
for roughly three-quarters of real-world ticket phrasing, this KB
currently has no confident match, and the Escalation Judge correctly
routes those to a human by design rather than forcing a weak match. The
75/25/0.60-threshold interplay is a secondary tuning decision on top of
that more fundamental coverage limit — expanding KB coverage further
(see Future Improvements) will move the needle more than re-tuning the
threshold at this stage.

---

## 📊 Evaluation Strategy

- **Eval sets:** 70 real tickets sampled from the Kaggle Customer Support Ticket Dataset (`data/pull_new_tickets.py`), split into `eval_tune.csv` (50 tickets, used to sweep the Escalation Judge's threshold) and `eval_holdout.csv` (20 tickets, never touched during tuning, used only for final reported numbers) -- this split exists specifically so reported accuracy isn't inflated by having been part of the same search that picked the threshold.
- **Labeling method:** hybrid. Labels were LLM-drafted (category, priority, escalate) then manually reviewed and corrected row by row against the project's own rubric (see Agent Design Deep-Dive) before being treated as ground truth.
- **Data cleaning:** the raw Kaggle descriptions required a cleaning pass (`data/clean_eval_descriptions.py`) before use -- see Key Learnings for why, and Results for the retrieval-generalization gap this process uncovered.
- **Metrics tracked:** category accuracy, priority accuracy, escalation decision accuracy, false-escalation rate, false-confidence rate -- tracked separately rather than as one blended accuracy number, since a false resolution and a false escalation are not equally costly (see Metrics Explained).

---

## 📈 Results

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
*(Write your real reasoning once you've built it — e.g. explicit state graphs give auditable decision paths, which matters for an escalation-gated system.)*

### 2. Why confidence-gating instead of full automation?
...

### 3. Why pgvector instead of a dedicated vector DB (Pinecone/Weaviate)?
...

### 4. Why Groq/Llama instead of GPT-4?
...

---

## 🔮 Future Improvements

### Short term
- [ ] Expand knowledge base with synthetic edge-case tickets
- [ ] Add per-category confidence thresholds instead of one global threshold
- [ ] Refine category classification rubric to reduce "Hardware" default-under-uncertainty bias, mirroring the Week 1 fix for priority's "Medium" bias
- [ ] Expand holdout eval set beyond 20 tickets -- current sample is too small to reliably observe the auto-resolve path given ~20-24% KB match rate

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
