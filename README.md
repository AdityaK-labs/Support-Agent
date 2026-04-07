---
title: MetaAI Support Agent
emoji: 🎧
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---
# OpenEnv-Based Autonomous Customer Support Agent

## Overview and Motivation
The **OpenEnv Customer Support Agent** simulates a real-world Business Process Outsourcing (BPO) and customer support environment. Standard generic benchmarks often use toy environments (like `echo` or grid grids); this environment tests the ability of an LLM agent to accurately read, comprehend, and resolve complex customer tickets autonomously. 

Agents are evaluated based on their ability to:
- Properly classify a ticket issue (e.g. shipping, billing, technical).
- Assign to the appropriate specialized team.
- Formulate an adequate response incorporating necessary details.
- Decide correctly when to issue a refund or when to escalate to management.

## Environment Specifications

### Observation Space
The observation provides the state of the ticket to the agent at every step.
- `ticket_id` (string): Unique identifier.
- `issue_type` (string): Category of the problem.
- `sentiment` (string): Customer's mood (positive, neutral, negative, angry).
- `priority` (string): Urgency (low, medium, high, critical).
- `message` (string): The customer's actual inquiry or complaint.
- `history` (list[string]): History of previous interactions/actions completed on this ticket.

### Action Space
The agent can execute multiple types of actions, submitted as strict JSON:
- `action_type` (string) [REQUIRED]: One of `classify`, `assign`, `respond`, `refund`, or `escalate`.
- `team` (string) [OPTIONAL]: The target team (e.g. `logistics_team`, `finance_team`). Required if action is `assign` or `escalate`.
- `response` (string) [OPTIONAL]: The text message to send back to the user. Required if action is `respond`, `refund`, or `escalate`.

### Reward Structure
Rewards are in the range of `[0.0, 1.0]` and calculated deterministically:
- **Classification / Base Action**: +0.3
- **Team Assignment**: +0.3
- **Response Quality** (Keyword Matching): +0.4 (Scaled)

**Penalties:**
- **Unnecessary refund**: -0.5
- **Unnecessary escalation**: -0.3
- **Excessive Steps**: -0.1 per step beyond 3.

## Tasks and Difficulties
There are 3 main tasks defining the difficulty of the episodes, governed by the `openenv.yaml`:
1. **Easy (Classification)**: The agent must simply classify the ticket to the right issue. Minimal steps.
2. **Medium (Assignment)**: The ticket must be correctly categorized and assigned to a specific internal department.
3. **Hard (Full Resolution)**: The agent must synthesize the context to decide if it should respond, issue a refund, or escalate directly to management—while generating a high-quality response.

## Setup Instructions
The environment strictly complies with the `OpenEnv` spec.

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Configure environment variables:**
   Place your keys inside `.env`:
   ```env
   HF_TOKEN=your_huggingface_token_here
   SUPABASE_URL=...
   ```
3. **Local run (Web UI):**
   ```bash
   python app.py
   ```
   *UI available on port 7860.*

## Containerized Execution (HF Spaces)
The submission includes a rigorous `Dockerfile` designed for Hugging Face Spaces. It binds to the standard `7860` port.
   ```bash
   docker build -t openenv-support-agent .
   docker run -p 7860:7860 openenv-support-agent
   ```

## Baseline Evaluation
A mandatory baseline inference script is provided inside the project root: `inference.py`.

Run baseline:
```bash
SUPPORT_ENV_TASK=hard python inference.py
```
This loops the specified agent through `MAX_STEPS` and prints standard stdout metrics:
```text
[START] task=hard env=support_env model=gpt-4o-mini
[STEP] step=1 action={"action_type": "escalate", ...} reward=1.00 done=true error=null
[END] success=true steps=1 score=1.000 rewards=1.00
```

## Environment Dynamics

The environment implements the full OpenEnv lifecycle:

- reset(): Initializes a new ticket scenario.
- state(): Returns the current environment state.
- step(action):
  - Validates action format
  - Simulates real-world outcome
  - Updates internal state
  - Computes reward using deterministic grader
  - Returns (observation, reward, done, info)

Episodes terminate when:
- Task objective is completed
- Maximum steps are exceeded

### Reward Design Philosophy

The reward function is shaped to reflect real-world business objectives:

- Encourages correct classification and routing
- Rewards precise and context-aware responses
- Penalizes costly or unnecessary actions (refunds, escalations)
- Discourages inefficient multi-step reasoning

The reward function is:
- Dense (provides intermediate feedback)
- Deterministic (same input → same output)
- Interpretable (clear contribution of each component)

## Known Limitations

- LLM-based agent performance depends on API availability and quota.
- To ensure robustness, a fallback deterministic policy is implemented.
- Future work includes reinforcement learning fine-tuning for improved decision quality.
