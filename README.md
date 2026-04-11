---
title: MetaAI Support Agent
emoji: 🎧
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# OpenEnv Autonomous Customer Support Agent

**Meta PyTorch Hackathon x Scaler School of Technology — Round 1**

An LLM-powered autonomous agent that resolves customer support tickets inside a fully OpenEnv-compliant RL environment. The agent reads ticket context (sentiment, priority, issue type, message history) and selects the single best action from a structured action space — classify, assign, respond, refund, or escalate — scored by a deterministic grader.

Live demo: [HuggingFace Space](https://huggingface.co/spaces/Tarun21W/MetaAI) | Source: [GitHub](https://github.com/AdityaK-labs/Support-Agent/tree/Tarun)

---

## Technical Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        HuggingFace Space (Docker)                   │
│                                                                     │
│  ┌──────────────┐     ┌──────────────────────────────────────────┐  │
│  │  Gradio UI   │────▶│           FastAPI App (app.py)           │  │
│  │  (port 7860) │     │  /health  /metadata  /schema  /mcp      │  │
│  └──────────────┘     │  /reset   /step      /state             │  │
│                       └───────────────┬──────────────────────────┘  │
│                                       │                             │
│                       ┌───────────────▼──────────────────────────┐  │
│                       │          SupportEnv (OpenEnv)            │  │
│                       │  reset() ─▶ Observation                  │  │
│                       │  step(Action) ─▶ (obs, reward, done)     │  │
│                       │  GraderEngine ─▶ proportional score      │  │
│                       └───────────────┬──────────────────────────┘  │
│                                       │                             │
└───────────────────────────────────────┼─────────────────────────────┘
                                        │
              ┌─────────────────────────▼──────────────────────────┐
              │             LiteLLM Proxy (Validator-injected)      │
              │   API_BASE_URL + HF_TOKEN  ──▶  LLM API call       │
              │   Model: Qwen/Qwen2.5-72B-Instruct (default)       │
              └────────────────────────────────────────────────────┘

inference.py (standalone evaluation):
  for task in [easy, medium, hard]:
      reset() ──▶ step loop (max 5) ──▶ [START]/[STEP]/[END] logs
```

---

## Environment Design

### Observation Space

Each step the agent receives a ticket object with:

| Field        | Type            | Description                                     |
|-------------|-----------------|------------------------------------------------|
| `ticket_id`  | string          | Unique ticket identifier (e.g. `TKT-H003`)     |
| `issue_type` | string          | Category: shipping, billing, technical, etc.    |
| `sentiment`  | string          | Customer mood: positive / neutral / negative / angry |
| `priority`   | string          | Urgency: low / medium / high / critical         |
| `message`    | string          | The raw customer message                        |
| `history`    | list\[string\]  | Previous actions taken in this episode          |

### Action Space

Actions are submitted as strict JSON objects:

| Field         | Type   | Required                          | Description                          |
|--------------|--------|-----------------------------------|--------------------------------------|
| `action_type` | string | Always                            | One of: classify, assign, respond, refund, escalate |
| `team`        | string | For assign / escalate             | Target specialist team               |
| `response`    | string | For respond / refund / escalate   | Customer-facing reply text           |

**Valid teams:**

| Team Name           | Handles                                                   |
|--------------------|-----------------------------------------------------------|
| `logistics_team`    | Lost packages, shipping tracking, delivery disputes       |
| `tech_support_team` | Login issues, password resets, bugs, API errors, data loss|
| `safety_team`       | Product defects, overheating, recalls, safety hazards     |
| `finance_team`      | Invoice errors, billing corrections, tax/payment issues   |
| `orders_team`       | Bulk orders, corporate accounts, order modifications      |
| `management_team`   | Escalated complaints, refund delays, manager requests     |

---

## Task Levels

The environment defines three difficulty tiers, each testing a different reasoning capability:

### Easy — Classification Only
The agent must output `{"action_type": "classify"}`. The grader checks only action type correctness. Any other action scores the minimum.

**Scenario examples:** address changes, account category updates, general unknown issues.

### Medium — Team Assignment
The agent must output `{"action_type": "assign", "team": "<correct_team>"}`. The grader scores both action type (+0.3) and team accuracy (+0.3), normalized to a max of 1.0.

**Scenario examples:**
- Missing package → `logistics_team`
- Product overheating → `safety_team`
- Can't log in → `tech_support_team`
- Wrong invoice name → `finance_team`
- Bulk order inquiry → `orders_team`

### Hard — Full Resolution
The agent must choose between escalate, refund, and respond — selecting the correct action and generating a quality response with specific keyword coverage.

**Scenario examples:**
- Manager demand + refund delay → `escalate` + `management_team`
- Wrong product received → `refund`
- API rate limit question → `respond` with technical details
- Data loss emergency → `escalate` + `tech_support_team`

---

## Reward System

The `GraderEngine` applies a normalized proportional scoring model:

### Score Components

```
max_possible = sum of components present in ground truth

Component             Points    Applied when
─────────────────────────────────────────────────────────
Correct action_type   +0.30     Always
Correct team          +0.30     Only if ground truth has a team
Keyword coverage      +0.40     Only if ground truth has response keywords
                                (scaled by fraction of keywords matched)

proportional_score = raw_score / max_possible
```

### Penalties (applied post-normalization)

| Violation                       | Penalty |
|--------------------------------|---------|
| Unnecessary refund              | −0.50   |
| Unnecessary escalation          | −0.30   |
| Wrong team assigned             | −0.15   |
| Each step beyond 3              | −0.10   |

### Final Score Formula

```python
final_score = clamp(proportional_score - penalties, 0.002, 0.998)
```

Scores are bounded to `[0.002, 0.998]` — 0.998 is a perfect score ceiling, 0.002 is the floor for failed episodes.

### Score Interpretation

| Score Range  | Interpretation                                       |
|-------------|------------------------------------------------------|
| 0.90 – 0.998 | Excellent: correct action + team + full keyword match |
| 0.60 – 0.89  | Good: correct action + team, partial response         |
| 0.30 – 0.59  | Partial: correct action type only                     |
| 0.002 – 0.29 | Poor: wrong action, or unnecessary penalty incurred   |

---

## Benchmark Results

Expected agent performance per task level (Qwen2.5-72B-Instruct with task-level instructions):

| Task    | Expected Score | Notes                                               |
|--------|---------------|-----------------------------------------------------|
| Easy   | ~0.95 – 0.998  | classify-only; model follows instruction reliably   |
| Medium | ~0.70 – 0.90   | team selection accuracy varies by domain keyword    |
| Hard   | ~0.50 – 0.85   | depends on response keyword coverage and action choice |

The agent runs up to 5 steps per episode. Episodes typically complete in 1 step for easy/medium tasks, and 1–2 steps for hard tasks.

---

## Agent Strategy

The agent uses a two-layer prompt:

1. **System prompt** — full SOP with action rules and penalty warnings
2. **User prompt** — task-level instruction injected per difficulty tier, current ticket JSON, recent history

Task-level instructions override the general SOP to prevent cross-task confusion (e.g., preventing `assign` on easy tasks or `classify` on hard tasks).

A fallback policy activates when the LLM call fails or returns unparseable JSON, ensuring the episode always completes without crashing.

---

## API Reference

All endpoints available at the Space URL:

| Method | Path         | Description                                      |
|-------|-------------|--------------------------------------------------|
| GET   | `/health`    | Returns `{"status": "healthy"}`                  |
| GET   | `/metadata`  | Environment name, description, tasks list        |
| GET   | `/schema`    | JSON schema for action/observation/state objects |
| POST  | `/mcp`       | MCP-compatible tool manifest                     |
| GET   | `/state`     | Current environment state                        |
| POST  | `/reset`     | Reset episode; accepts `?task_name=easy/medium/hard` |
| POST  | `/step`      | Submit an action; body: `ActionRequest` JSON     |

---

## Project Structure

```
meta/
├── app.py                   # FastAPI + Gradio UI (root entry point)
├── inference.py             # Standalone evaluation script (runs all 3 tasks)
├── openenv.yaml             # OpenEnv spec: tasks, graders, reward range
├── pyproject.toml           # Package config + openenv-core dependency
├── requirements.txt         # Runtime dependencies
├── Dockerfile               # HF Spaces Docker build
├── server/
│   └── app.py               # Mirror of app.py (required by openenv validate)
├── openenv/
│   ├── env.py               # SupportEnv: reset/step/state logic
│   ├── models.py            # Pydantic models: Action, Observation, Reward, etc.
│   ├── graders/
│   │   └── grader.py        # GraderEngine: deterministic scoring
│   └── tasks/
│       ├── easy.py          # 5 easy ticket scenarios
│       ├── medium.py        # 5 medium ticket scenarios
│       └── hard.py          # 5 hard ticket scenarios
└── support_env.py           # SupportEnvWrapper for inference.py (Docker-based)
```

---

## Setup & Deployment

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export API_BASE_URL=https://your-llm-proxy-url/v1
export HF_TOKEN=your_token_here

# Run the UI
python app.py
# → http://localhost:7860
```

### Run Inference Script

```bash
python inference.py
# Runs all 3 tasks and emits structured logs:
# [START] task=easy env=support_env model=Qwen/Qwen2.5-72B-Instruct
# [STEP] step=1 action={"action_type":"classify"} reward=1.00 done=true error=null
# [END] success=true steps=1 score=0.998 rewards=1.00
```

### Docker Build

```bash
docker build -t openenv-support-agent .
docker run -p 7860:7860 \
  -e API_BASE_URL=https://your-proxy/v1 \
  -e HF_TOKEN=your_token \
  openenv-support-agent
```

### HuggingFace Spaces Deployment

1. Push to the Space's git remote
2. Set secrets in **Settings → Repository secrets**: `API_BASE_URL`, `HF_TOKEN`
3. The Space builds automatically from `Dockerfile`

---

## Environment Variables

| Variable       | Required | Description                                              |
|---------------|----------|----------------------------------------------------------|
| `API_BASE_URL` | Yes      | Base URL of the OpenAI-compatible LLM proxy              |
| `HF_TOKEN`     | Yes      | HuggingFace token (primary credential for LLM proxy)     |
| `API_KEY`      | Fallback  | Alternative API key if `HF_TOKEN` is not set            |
| `MODEL_NAME`   | No       | LLM model name (default: `Qwen/Qwen2.5-72B-Instruct`)   |
| `PORT`         | No       | Server port (default: `7860`)                            |

---

## Roadmap

- **RL fine-tuning**: Use episode trajectories to fine-tune a smaller model (e.g. Qwen2.5-7B) via PPO/GRPO on the GraderEngine reward signal
- **Multi-turn episodes**: Extend action space to allow follow-up questions before resolution
- **Ticket generator**: Procedurally generate tickets with controllable difficulty and domain distribution
- **Confidence calibration**: Add a confidence field to actions; penalize low-confidence wrong answers more heavily
- **Leaderboard**: Track model performance across runs with per-task score history

---

## Reward Design Philosophy

The reward function mirrors real-world BPO business objectives:

- **Dense feedback** — every step returns an interpretable score with per-component breakdown
- **Deterministic** — same ticket + action always yields the same score; no randomness
- **Proportional normalization** — easy tasks (classify only) can achieve 1.0 without being penalized for missing team/response components that were never required
- **Penalty asymmetry** — refund mistakes (−0.50) cost more than escalation mistakes (−0.30) reflecting real business cost; wrong team routing (−0.15) is a softer penalty since it still shows intent