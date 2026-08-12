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

This project is a multi-agent system that automates first-line IT support ticket handling. Given an incoming ticket (subject, description, product/category context), the system classifies it, checks for a known resolution, attempts an autonomous fix via tool calls, and — critically — **knows when not to act**, escalating to a human agent when its own confidence is low.

**Final Result: [X]% resolution accuracy, [Y]% false-escalation rate, [Z]% false-confidence rate on a [N]-ticket held-out eval set.**

*(Fill this in once Week 3 eval is done — this single line is what a recruiter reads first.)*

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

*(Replace this ASCII diagram with a screenshot of your actual LangGraph state graph once built — `langgraph` can export one directly.)*

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Backend | FastAPI, Uvicorn, Pydantic v2, SQLAlchemy |
| Database | PostgreSQL (Neon) |
| Vector store | pgvector |
| Agent orchestration | LangGraph, LangChain |
| LLM inference | Groq API (Llama 3.3 70B) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Evaluation | pandas, scikit-learn |
| Dashboard | Streamlit |
| Deployment | Render |
| CI | GitHub Actions |

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

---

## ⚙️ Installation

```bash
git clone https://github.com/<yourusername>/it-triage-agent.git
cd it-triage-agent

pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add GROQ_API_KEY, DATABASE_URL

# Run database migrations
python -m app.db.init

# Start the API
uvicorn app.main:app --reload
```

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

*(Fill this in as you build each agent — mirror the "Feature Engineering" depth from the reference project. For each agent, explain: what it takes as input, how it decides, and one concrete example.)*

### Classifier Agent
- Input: ...
- Decision logic: ...
- Example: ...

### Retriever Agent
- Similarity threshold used: ...
- What happens on no match: ...

### Resolver Agent
- Tools available: ...
- When it chooses to act vs draft-only: ...

---

## 🚦 Escalation Logic

*(This is your project's signature section — treat it like the reference repo's "LoRA Configuration" deep-dive.)*

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

| Metric | Value |
|---|---|
| Resolution accuracy | [X]% |
| False-escalation rate | [Y]% |
| False-confidence rate | [Z]% |
| Avg. response latency | [T]s |

### Per-category performance

| Category | Accuracy | Status |
|---|---|---|
| Password/Account | XX% | ✅ Strong |
| Network | XX% | ⚠️ Moderate |
| Hardware | XX% | ⚠️ Moderate |
| Billing | XX% | ❌ Weak — [explain why] |

*(Being honest about a weak category, like the reference repo did with high-price SMAPE, is what makes this credible instead of marketing copy.)*

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

**Built with FastAPI · LangGraph · Groq · pgvector**# ConfidenceGatedTriageAgent
