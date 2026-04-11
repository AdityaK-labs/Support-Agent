---
title: MetaAI Support Agent
emoji: 🎧
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

```
 ██████╗ ██████╗ ███████╗███╗   ██╗███████╗███╗   ██╗██╗   ██╗
██╔═══██╗██╔══██╗██╔════╝████╗  ██║██╔════╝████╗  ██║██║   ██║
██║   ██║██████╔╝█████╗  ██╔██╗ ██║█████╗  ██╔██╗ ██║╚██╗ ██╔╝
██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██╔══╝  ██║╚██╗██║ ╚████╔╝
╚██████╔╝██║     ███████╗██║ ╚████║███████╗██║ ╚████║  ╚██╔╝
 ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═══╝   ╚═╝
```

# OpenEnv Autonomous Customer Support Agent

**Meta PyTorch Hackathon x Scaler School of Technology — Round 1**

An LLM-powered autonomous agent that resolves customer support tickets inside a fully OpenEnv-compliant reinforcement learning environment. Each episode is a **3-phase Markov Decision Process** — the agent must triage, route, and resolve tickets through sequential decision-making, with dense intermediate rewards shaping behavior at every step.

**Live Demo:** [huggingface.co/spaces/Tarun21W/MetaAI](https://huggingface.co/spaces/Tarun21W/MetaAI) &nbsp;|&nbsp; **Source:** [github.com/AdityaK-labs/Support-Agent](https://github.com/AdityaK-labs/Support-Agent/tree/Tarun)

---

## What Makes This a Real RL Problem

Most LLM-as-agent benchmarks treat each ticket as a single-step classification task — a bandit problem, not reinforcement learning. This environment is different:

- **State evolves between steps.** After Phase 1, the true `issue_type` is revealed in the observation. After Phase 2, the assigned team appears in history. The agent sees a richer state at each step.
- **Actions have consequences.** A wrong team assignment in Phase 2 directly reduces Phase 3 quality — the agent cannot "undo" routing decisions.
- **Dense rewards, not sparse.** Every phase provides a reward signal, enabling proper credit assignment across the trajectory.
- **Partial recovery is possible.** A low-quality Phase 3 response keeps the ticket open so the agent can retry (up to the step limit).

---

## Technical Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                       HuggingFace Space (Docker)                     │
│                                                                      │
│   ┌─────────────────┐      ┌────────────────────────────────────┐   │
│   │   Gradio UI      │─────▶│        FastAPI (app.py)            │   │
│   │  3-phase display │      │  /health /metadata /schema /mcp   │   │
│   └─────────────────┘      │  /reset  /step     /state          │   │
│                             └───────────────┬────────────────────┘   │
│                                             │                        │
│                             ┌───────────────▼────────────────────┐   │
│                             │        SupportEnv (OpenEnv)        │   │
│                             │                                    │   │
│                             │  reset() ──▶ Observation (phase 1) │   │
│                             │                                    │   │
│                             │  step(classify)                    │   │
│                             │    ──▶ reveal issue_type (phase 2) │   │
│                             │                                    │   │
│                             │  step(assign + team)               │   │
│                             │    ──▶ add team context (phase 3)  │   │
│                             │                                    │   │
│                             │  step(escalate/refund/respond)     │   │
│                             │    ──▶ GraderEngine ──▶ reward     │   │
│                             │    ──▶ done = True                 │   │
│                             └────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘

inference.py (standalone validator):
  for task in [easy, medium, hard]:
      reset() → step×3 → [START]/[STEP]/[END] structured logs
```

---

## The 3-Phase MDP

Every episode — regardless of task difficulty — progresses through exactly three phases:

```
Phase 1: TRIAGE          Phase 2: ROUTE           Phase 3: RESOLVE
─────────────────        ──────────────────        ─────────────────────
Agent reads ticket   →   issue_type revealed   →   team context in obs
Agent classifies         Agent assigns team         Agent resolves ticket
reward: 0.998/0.002      reward: 0.998/0.4/0.002    reward: 0.002–0.998
```

### Phase 1 — Triage
The agent sees a raw ticket with `issue_type: unknown`. Its only job is to recognize that classification is needed:
```json
{"action_type": "classify"}
```
The environment then reveals the true `issue_type` in the observation for the next step.

### Phase 2 — Route
Now knowing the issue type, the agent routes the ticket to the correct specialist team:
```json
{"action_type": "assign", "team": "tech_support_team"}
```
The assigned team is appended to the observation history, informing the resolution phase.

### Phase 3 — Resolve
The agent takes the final action based on full context (ticket + revealed issue + routed team):

| Action | When to use | Required fields |
|--------|-------------|-----------------|
| `escalate` | Customer demands manager, safety hazard, or data emergency | `team` + `response` |
| `refund` | Product provably wrong/broken and company is at fault | `response` |
| `respond` | Customer needs information, pricing, or technical details | `response` |

---

## Observation Space

The agent's view evolves across phases:

| Field | Type | Phase 1 | Phase 2 | Phase 3 |
|-------|------|---------|---------|---------|
| `ticket_id` | string | ✓ | ✓ | ✓ |
| `issue_type` | string | `"unknown"` | revealed | revealed |
| `sentiment` | string | ✓ | ✓ | ✓ |
| `priority` | string | ✓ | ✓ | ✓ |
| `message` | string | ✓ | ✓ | ✓ |
| `history` | list | `[]` | classify result | classify + assign |

---

## Reward System

### Phase Rewards

**Phase 1 (Triage):** Binary
```
classify action  →  0.998
any other action →  0.002
```

**Phase 2 (Route):** Graded
```
assign + correct team  →  0.998
assign + wrong team    →  0.400
not an assign action   →  0.002
```

**Phase 3 (Resolve):** Proportional quality scoring
```
Component                      Max     Applied when
─────────────────────────────────────────────────────
Correct action type            +0.30   always
Correct team                   +0.30   if ground truth has a team
Response keyword coverage      +0.40   scaled by fraction matched

proportional_score = raw / max_possible
```

**Penalties (Phase 3 only):**

| Violation | Penalty |
|-----------|---------|
| Unnecessary refund | −0.50 |
| Unnecessary escalation | −0.30 |
| Wrong team assigned | −0.15 |
| Each step beyond 3 | −0.10 |

**Final episode score:**
```
step_score  = clamp(proportional - penalties, 0.002, 0.998)
episode_score = mean([phase1_reward, phase2_reward, phase3_reward])
```

### Score Interpretation

| Episode Score | Meaning |
|--------------|---------|
| 0.90 – 0.998 | Near-perfect: all 3 phases correct, full keyword coverage |
| 0.70 – 0.89 | Good: triage/route correct, partial response quality |
| 0.50 – 0.69 | Partial: triage correct, routing or resolution error |
| 0.002 – 0.49 | Poor: wrong action type or heavy penalty incurred |

---

## Task Levels

Three difficulty tiers vary the complexity of the resolution phase (Phase 3):

### Easy
Straightforward tickets with clear intent. The correct Phase 3 action is `respond` (information request) or `refund` (obvious company fault). No escalation needed.

**Example scenarios:**
- Wrong shipping address → respond with update confirmation
- Double charge on credit card → refund with apology
- Can't update profile picture → respond with steps

### Medium
Tickets requiring domain knowledge to route correctly in Phase 2, plus confident Phase 3 resolution.

**Example scenarios:**
- Missing package (tracking says delivered) → logistics_team → respond
- Product overheating / fire hazard → safety_team → **escalate** (safety emergency)
- Password reset link not arriving → tech_support_team → respond
- Wrong company name on invoice → finance_team → respond
- Bulk discount inquiry → orders_team → respond

### Hard
Ambiguous tickets where the agent must distinguish between escalate, refund, and respond — including understanding when manager demands or safety emergencies override normal flow.

**Example scenarios:**
- 3-week refund delay + manager demand → management_team → escalate
- Wrong product delivered (red vs blue) → orders_team → refund
- API 429 rate limit question → tech_support_team → respond with technical detail
- Data deleted by software bug → tech_support_team → escalate (emergency)

---

## Team Directory

| Team | Handles |
|------|---------|
| `logistics_team` | Lost packages, shipping tracking, delivery disputes |
| `tech_support_team` | Login issues, password resets, bugs, API errors, data loss |
| `safety_team` | Product defects, overheating, recalls, safety hazards |
| `finance_team` | Invoice errors, billing corrections, tax/payment issues |
| `orders_team` | Bulk orders, corporate accounts, returns, subscriptions |
| `management_team` | Escalated complaints, refund delays, manager requests |

---

## Benchmark Performance

Expected agent performance using `Qwen/Qwen2.5-72B-Instruct` with phase-aware prompting:

| Task | Phase 1 | Phase 2 | Phase 3 | Avg Episode Score |
|------|---------|---------|---------|-------------------|
| Easy | ~0.998 | ~0.90 | ~0.70 | ~0.87 |
| Medium | ~0.998 | ~0.85 | ~0.65 | ~0.83 |
| Hard | ~0.998 | ~0.80 | ~0.60 | ~0.80 |

Phase 1 is near-perfect because the instruction is unambiguous. Phase 2 depends on domain keyword matching. Phase 3 is the hardest — response quality and action selection under ambiguity determine the final score.

---

## Agent Design

The agent uses a layered prompt strategy:

**System prompt** — defines the 3-phase structure, all action rules, and penalty warnings.

**Per-step user prompt** — includes the current phase instruction, live ticket JSON, and the last 6 history entries:

```
CURRENT PHASE: 2 — ROUTE
The issue type is now known (see history). Assign to the correct team.
Output: {"action_type": "assign", "team": "<team_name>"}

Ticket: {
  "ticket_id": "TKT-M003",
  "issue_type": "technical",       ← revealed by Phase 1
  ...
}
History:
  [Phase 1/triage] Agent: classify
  System: Issue classified as 'technical'. Proceed to route the ticket.
```

**Fallback policy** — if the LLM call fails or returns unparseable JSON, a phase-appropriate deterministic fallback activates so the episode always completes.

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Returns `{"status": "healthy"}` |
| GET | `/metadata` | Environment name, description, supported tasks |
| GET | `/schema` | JSON schema for action / observation / state |
| POST | `/mcp` | MCP-compatible tool manifest |
| GET | `/state` | Current environment state (phase, step count, history) |
| POST | `/reset?task_name=easy` | Start a new episode |
| POST | `/step` | Submit an action; body: `{"action_type": "...", "team": "...", "response": "..."}` |

---

## Project Structure

```
meta/
├── app.py                      # FastAPI + Gradio UI (root entry point)
├── inference.py                # Standalone evaluation (runs all 3 tasks)
├── openenv.yaml                # OpenEnv spec: tasks, graders, reward range [0.002, 0.998]
├── pyproject.toml              # Package config + openenv-core dependency
├── requirements.txt            # Runtime dependencies
├── Dockerfile                  # HuggingFace Spaces build
├── server/
│   └── app.py                  # Mirror of app.py (required by openenv validate)
└── openenv/
    ├── env.py                  # SupportEnv: 3-phase MDP logic
    ├── models.py               # Pydantic models: Action, Observation, Reward, etc.
    ├── reward.py               # Dispatch to GraderEngine with phase context
    ├── graders/
    │   └── grader.py           # GraderEngine: phase-aware + proportional scoring
    └── tasks/
        ├── easy.py             # 5 easy ticket scenarios (3-phase)
        ├── medium.py           # 5 medium ticket scenarios (3-phase)
        └── hard.py             # 5 hard ticket scenarios (3-phase)
```

---

## Setup & Deployment

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
export API_BASE_URL=https://your-llm-proxy/v1
export HF_TOKEN=your_token_here

# Launch UI + API
python app.py
# → http://localhost:7860
```

### Run Inference (All 3 Tasks)

```bash
python inference.py
```

Output format:
```
[INFO] Using API_BASE_URL=https://... MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
[START] task=easy env=support_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"action_type":"classify"} reward=1.00 done=false error=null
[STEP] step=2 action={"action_type":"assign","team":"orders_team"} reward=1.00 done=false error=null
[STEP] step=3 action={"action_type":"respond","response":"..."} reward=0.87 done=true error=null
[END] success=true steps=3 score=0.957 rewards=1.00,1.00,0.87
```

### Docker Build

```bash
docker build -t openenv-support-agent .
docker run -p 7860:7860 \
  -e API_BASE_URL=https://your-proxy/v1 \
  -e HF_TOKEN=your_token \
  openenv-support-agent
```

### HuggingFace Spaces

Push to the Space's git remote. Set secrets in **Settings → Repository secrets**:
- `API_BASE_URL` — LLM proxy base URL
- `HF_TOKEN` — your HuggingFace token

The Space builds automatically from `Dockerfile` and binds to port `7860`.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | Yes | Base URL of the OpenAI-compatible LLM proxy |
| `HF_TOKEN` | Yes | Primary credential for the LLM proxy |
| `API_KEY` | Fallback | Used if `HF_TOKEN` is not set |
| `MODEL_NAME` | No | LLM model name (default: `Qwen/Qwen2.5-72B-Instruct`) |
| `PORT` | No | Server port (default: `7860`) |

---

## Reward Design Philosophy

The reward function is built around four real-world BPO principles:

**1. Dense feedback** — every step returns an interpretable score. Agents are never left guessing what went wrong across a long episode.

**2. Proportional normalization** — scores are normalized by what the scenario actually tests. An easy ticket that only needs `classify` can still achieve `0.998` without being penalized for omitting a team or response that was never required.

**3. Asymmetric penalties** — refund mistakes (`−0.50`) cost more than escalation mistakes (`−0.30`), which cost more than wrong team (`−0.15`). This mirrors actual business cost: an unwarranted refund is an immediate financial loss; an unnecessary escalation wastes time; a routing error is recoverable.

**4. Deterministic scoring** — the same action on the same ticket always produces the same reward. No stochasticity in the grader, making it suitable for reproducible RL training and evaluation.

---

## Roadmap

- **RL fine-tuning**: Collect episode trajectories and use PPO/GRPO to fine-tune Qwen2.5-7B directly on the GraderEngine reward signal
- **Multi-turn Phase 3**: Allow the agent to ask clarifying questions before resolving — rewarding efficient resolution over blind action
- **Procedural ticket generator**: Generate tickets with controllable sentiment, domain, and ambiguity for richer training distributions
- **Confidence-gated routing**: Add a `confidence` field; low-confidence phase-3 actions trigger human review before closing the ticket
- **Cross-model leaderboard**: Compare Qwen, LLaMA, Mistral, and GPT families on the same task distribution with per-phase score breakdown