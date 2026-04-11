import asyncio
import json
import os
import re
import textwrap
from typing import List, Optional

import gradio as gr
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv()

from openenv.env import SupportEnv
from openenv.models import Action
from openenv.tasks import TASK_REGISTRY

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

TEMPERATURE = 0.2
MAX_TOKENS  = 500

# ---------------------------------------------------------------------------
# FastAPI app + environment
# ---------------------------------------------------------------------------

app = FastAPI(title="OpenEnv Support Agent API")
env = SupportEnv(task_name="easy")
env.reset()

class ActionRequest(BaseModel):
    action_type: str
    team: str = None
    response: str = None

@app.get("/api/state")
@app.get("/state")
def get_state():
    return env.state()

@app.post("/api/reset")
@app.post("/reset")
def reset_env(task_name: str = "easy"):
    return env.reset(task_name=task_name)

@app.post("/api/step")
@app.post("/step")
def step_env(action_req: ActionRequest):
    act = Action(**action_req.model_dump())
    return env.step(act)

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/metadata")
def metadata():
    return {
        "name": "customer-support-agent",
        "description": (
            "OpenEnv-compliant autonomous customer support environment. "
            "An AI agent processes BPO/customer-support tickets deciding "
            "classify, assign, respond, refund, or escalate actions."
        ),
        "version": "1.0.0",
        "tasks": ["easy", "medium", "hard"],
    }

@app.get("/schema")
def schema():
    return {
        "action": {
            "type": "object",
            "properties": {
                "action_type": {"type": "string", "enum": ["classify", "assign", "respond", "refund", "escalate"]},
                "team":        {"type": "string", "nullable": True},
                "response":    {"type": "string", "nullable": True},
            },
            "required": ["action_type"],
        },
        "observation": {
            "type": "object",
            "properties": {
                "ticket_id":  {"type": "string"},
                "issue_type": {"type": "string"},
                "sentiment":  {"type": "string"},
                "priority":   {"type": "string"},
                "message":    {"type": "string"},
                "history":    {"type": "array", "items": {"type": "string"}},
            },
        },
        "state": {
            "type": "object",
            "properties": {
                "observation":   {"type": "object"},
                "step_count":    {"type": "integer"},
                "done":          {"type": "boolean"},
                "episode_id":    {"type": "string"},
                "total_reward":  {"type": "number"},
                "task_name":     {"type": "string"},
                "max_steps":     {"type": "integer"},
            },
        },
    }

@app.post("/mcp")
def mcp(request: dict = None):
    return {
        "jsonrpc": "2.0",
        "result": {
            "tools": [
                {"name": "reset", "description": "Reset the environment"},
                {"name": "step",  "description": "Take a step in the environment"},
                {"name": "state", "description": "Get current environment state"},
            ]
        },
        "id": None,
    }

# ---------------------------------------------------------------------------
# Agent LLM logic
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are an autonomous customer support agent for a large e-commerce and SaaS company.

    EVERY episode runs through exactly 3 phases. Complete each phase in order:

    PHASE 1 — TRIAGE: Classify the ticket issue type.
    Action: {"action_type": "classify"}

    PHASE 2 — ROUTE: Assign to the correct specialist team.
    Action: {"action_type": "assign", "team": "<team_name>"}
    Teams: logistics_team | tech_support_team | safety_team | finance_team | orders_team | management_team

    PHASE 3 — RESOLVE: Take the final resolution action.
      escalate — angry/manager request/safety emergency (include team + response). PENALTY -0.30 if unnecessary.
      refund   — provably company fault/wrong product (include response). PENALTY -0.50 if unnecessary.
      respond  — customer needs information; write a detailed specific response.

    OUTPUT FORMAT — MANDATORY:
    Raw JSON only. No markdown, no explanation.
    {"action_type": "...", "team": "...", "response": "..."}
""").strip()

PHASE_PROMPTS = {
    1: (
        "CURRENT PHASE: 1 — TRIAGE\n"
        "Your task: Classify this ticket.\n"
        "Output exactly: {\"action_type\": \"classify\"}"
    ),
    2: (
        "CURRENT PHASE: 2 — ROUTE\n"
        "The issue type is now known (see history). Assign to the correct team.\n"
        "Output: {\"action_type\": \"assign\", \"team\": \"<team_name>\"}"
    ),
    3: (
        "CURRENT PHASE: 3 — RESOLVE\n"
        "Choose the correct final action: escalate / refund / respond.\n"
        "Include team if escalating. Include a detailed response for respond/refund/escalate."
    ),
}


def _build_prompt(obs, history: List[str], phase: int = 1) -> str:
    ctx = {
        "ticket_id":  obs.ticket_id,
        "issue_type": obs.issue_type,
        "sentiment":  obs.sentiment,
        "priority":   obs.priority,
        "message":    obs.message,
    }
    hist = "\n".join(history[-6:]) if history else "None"
    instruction = PHASE_PROMPTS.get(phase, PHASE_PROMPTS[3])
    return textwrap.dedent(f"""
        {instruction}

        Ticket: {json.dumps(ctx, indent=2)}
        History:
        {hist}
        Respond with JSON only.
    """).strip()


async def _call_llm(obs, history: List[str], phase: int = 1) -> dict:
    if not API_BASE_URL or not API_KEY:
        raise RuntimeError("API_BASE_URL / HF_TOKEN not configured.")
    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    prompt = _build_prompt(obs, history, phase)
    resp = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    text = resp.choices[0].message.content.strip()
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON in LLM response: {text}")
    return json.loads(match.group(1))

# ---------------------------------------------------------------------------
# UI helpers & HTML renderers
# ---------------------------------------------------------------------------

TASK_MAP = {
    "Easy (Classification)":  "easy",
    "Medium (Assignment)":    "medium",
    "Hard (Full Resolution)": "hard",
}

TEAM_DESCRIPTIONS = {
    "logistics_team":    "Logistics Team",
    "tech_support_team": "Tech Support Team",
    "safety_team":       "Safety Team",
    "finance_team":      "Finance Team",
    "orders_team":       "Orders Team",
    "management_team":   "Management Team",
}

TEAM_SUBTITLES = {
    "logistics_team":    "Shipping, delivery, lost packages",
    "tech_support_team": "Login, bugs, API errors, data loss",
    "safety_team":       "Defects, overheating, recalls, hazards",
    "finance_team":      "Invoices, billing, tax, payment issues",
    "orders_team":       "Bulk orders, returns, subscriptions",
    "management_team":   "Escalated complaints, manager requests",
}

_PHASE_COLORS  = {1: "#60a5fa", 2: "#ffa94d", 3: "#00d084"}
_PHASE_NAMES   = {1: "Triage", 2: "Route", 3: "Resolve"}
_TIER_COLORS   = {"easy": "#00d084", "medium": "#ffa94d", "hard": "#ff6b6b"}
_SENT_COLORS   = {"positive": "#00d084", "neutral": "#9ca3af", "negative": "#ffa94d", "angry": "#ff6b6b"}
_PRIO_COLORS   = {"low": "#9ca3af", "medium": "#60a5fa", "high": "#ffa94d", "critical": "#ff6b6b"}
_ACTION_COLORS = {"classify": "#60a5fa", "assign": "#ffa94d", "respond": "#00d084", "refund": "#c084fc", "escalate": "#ff6b6b"}

_agent_history: List[str] = []


def _esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def _score_color(s: float) -> str:
    if s >= 0.8:  return "#00d084"
    if s >= 0.5:  return "#ffa94d"
    return "#ff6b6b"

def _badge(label: str, color: str) -> str:
    return (
        f"<span style='display:inline-block;padding:2px 9px;border-radius:4px;"
        f"background:{color}22;color:{color};border:1px solid {color}66;"
        f"font-size:11px;font-weight:600;font-family:JetBrains Mono,monospace;"
        f"letter-spacing:.4px;text-transform:uppercase'>{_esc(label)}</span>"
    )


def _obs_html(obs_json: str) -> str:
    if not obs_json:
        return "<div class='pg-placeholder'>Reset to load a ticket.</div>"
    return f"<pre class='pg-code'>{_esc(obs_json)}</pre>"


def _steps_html(steps: List[str]) -> str:
    if not steps:
        return "<div class='pg-placeholder'>No steps taken yet.</div>"
    rows = ""
    for s in steps:
        color = "#9ca3af"
        for p, c in _PHASE_COLORS.items():
            if f"Phase {p}:" in s:
                color = c
                break
        rows += f"<div class='trace-step' style='border-left:3px solid {color};background:{color}0d'><code>{_esc(s)}</code></div>"
    return f"<div class='trace-wrap'>{rows}</div>"


def _score_html(rewards: List[float], avg: float, done: bool) -> str:
    if not rewards:
        return "<div class='pg-placeholder'>No scores yet.</div>"
    badges = " ".join(
        f"<span class='score-badge' style='background:{_score_color(r)}22;color:{_score_color(r)};border:1px solid {_score_color(r)}66'>{r:.3f}</span>"
        for r in rewards
    )
    ac  = _score_color(avg)
    dc  = "#00d084" if done else "#ffa94d"
    return f"""
<div class='pg-status'>
  <div class='stat-item'><span class='stat-label'>Steps</span><span class='stat-val'>{len(rewards)}</span></div>
  <div class='stat-item'><span class='stat-label'>Per-Phase Rewards</span><span>{badges}</span></div>
  <div class='stat-item'><span class='stat-label'>Avg Score</span><span class='stat-val' style='color:{ac};font-size:18px'>{avg:.3f}</span></div>
  <div class='stat-item'><span class='stat-label'>Done</span><span class='stat-val' style='color:{dc}'>{str(done).lower()}</span></div>
</div>"""


def _team_html(team_lines: List[str]) -> str:
    if not team_lines:
        return "<div class='pg-placeholder'>(no team assigned — triage or respond action)</div>"
    rows = ""
    for line in team_lines:
        # parse "[TKT-XXX]  Team Name — subtitle"
        rows += f"<div class='team-row'><code>{_esc(line)}</code></div>"
    return f"<div class='team-wrap'>{rows}</div>"


def _history_html() -> str:
    if not env.history:
        return "<div class='pg-placeholder'>(empty)</div>"
    rows = ""
    for entry in env.history:
        if entry.startswith("System:") or entry.startswith("Customer:"):
            cls = "hist-system"
        else:
            cls = "hist-agent"
        rows += f"<div class='hist-row {cls}'>{_esc(entry)}</div>"
    return f"<div class='hist-wrap'>{rows}</div>"


def _msg_html(msg: str) -> str:
    if not msg:
        return "<div class='pg-placeholder'>(no customer-facing message — triage or route phase)</div>"
    return f"<div class='agent-msg'>{_esc(msg)}</div>"


def _manual_score_html(reward: float, done: bool, feedback: str) -> str:
    c = _score_color(reward)
    dc = "#00d084" if done else "#ffa94d"
    fb = feedback[:300] + "..." if len(feedback) > 300 else feedback
    return f"""
<div class='pg-status'>
  <div class='stat-item'><span class='stat-label'>Score</span><span class='stat-val' style='color:{c};font-size:18px'>{reward:.3f}</span></div>
  <div class='stat-item'><span class='stat-label'>Done</span><span class='stat-val' style='color:{dc}'>{str(done).lower()}</span></div>
  <div class='stat-item' style='flex:2'><span class='stat-label'>Grader Feedback</span><code style='font-size:11px;color:#9ca3af'>{_esc(fb)}</code></div>
</div>"""

# ---------------------------------------------------------------------------
# Static HTML content sections
# ---------------------------------------------------------------------------

def _overview_html() -> str:
    cards = [
        ("⚡", "3-Phase MDP", "Not a bandit. Every ticket runs through Triage → Route → Resolve with state evolving between steps — a proper sequential decision problem."),
        ("🎯", "Dense Rewards", "Every phase returns an intermediate reward. No sparse end-of-episode signal. Phase 1 binary, Phase 2 graded, Phase 3 quality-scored."),
        ("🔀", "State Transitions", "After Phase 1 the observation's issue_type updates from 'unknown' to the true category. After Phase 2 the assigned team appears in history. The agent sees richer context at each step."),
        ("⚖️", "Asymmetric Penalties", "Refund (-0.50) > Escalation (-0.30) > Wrong Team (-0.15). Mirrors real business cost: an unwarranted refund is immediate financial loss."),
        ("📊", "Proportional Scoring", "Phase 3 normalizes raw_score / max_possible so a simple ticket can still score 0.998 without being penalized for omitting components it never needed."),
        ("🔒", "Deterministic Grader", "Same action + same ticket = same reward. No stochasticity in the grader. Reproducible RL training and evaluation."),
    ]
    grid = ""
    for icon, title, body in cards:
        grid += f"""
<div class='unique-card'>
  <div class='card-icon'>{icon}</div>
  <div class='card-title'>{title}</div>
  <div class='card-body'>{body}</div>
</div>"""
    arch = """
<div class='arch-box'>
<pre class='pg-code' style='margin:0'>Episode Flow (3 steps per ticket):

  reset()
    └─▶ Observation {issue_type: "unknown", sentiment, priority, message}

  step({"action_type": "classify"})          ← Phase 1: Triage
    └─▶ issue_type revealed in observation
    └─▶ reward: 0.998 (correct) or 0.002

  step({"action_type": "assign", "team": …}) ← Phase 2: Route
    └─▶ team context appended to history
    └─▶ reward: 0.998 / 0.400 / 0.002

  step({"action_type": "respond/refund/escalate", …}) ← Phase 3: Resolve
    └─▶ done = True
    └─▶ reward: proportional quality score in [0.002, 0.998]

episode_score = mean(phase1_reward, phase2_reward, phase3_reward)</pre>
</div>"""
    return f"""
<h2 class='section-h'>What makes this a real RL environment</h2>
<p class='section-sub'>Six design decisions that distinguish this from single-step LLM evaluation benchmarks.</p>
<div class='unique-grid'>{grid}</div>
<h2 class='section-h' style='margin-top:40px'>Episode Architecture</h2>
<p class='section-sub'>Each episode is a deterministic 3-step MDP. Observation state updates between steps.</p>
{arch}"""


def _scenarios_html() -> str:
    out = ""
    for tier in ["easy", "medium", "hard"]:
        scenarios = TASK_REGISTRY[tier]
        tc = _TIER_COLORS[tier]
        out += f"<h3 class='tier-header' style='color:{tc}'>{tier.upper()} TIER — {len(scenarios)} scenarios</h3>"
        for sc in scenarios:
            t = sc.ticket
            gt = sc.ground_truth
            sc_color = _ACTION_COLORS.get(gt.action_type, "#9ca3af")
            sent_c = _SENT_COLORS.get(t.sentiment, "#9ca3af")
            prio_c = _PRIO_COLORS.get(t.priority, "#9ca3af")
            kw_html = " ".join(
                f"<code class='kw-tag'>{_esc(k)}</code>"
                for k in gt.response_keywords
            ) if gt.response_keywords else "<span class='muted'>none</span>"
            team_html = (
                f"<code style='color:#ffa94d'>{_esc(gt.team)}</code>"
                if gt.team else "<span class='muted'>—</span>"
            )
            out += f"""
<div class='scen-card'>
  <div class='scen-header'>
    {_badge(tier, tc)}
    {_badge(t.sentiment, sent_c)}
    {_badge(t.priority, prio_c)}
    <code class='ticket-id'>{_esc(t.ticket_id)}</code>
  </div>
  <div class='scen-message'>&ldquo;{_esc(t.message)}&rdquo;</div>
  <div class='scen-meta'>
    <div class='meta-item'>
      <span class='meta-label'>Phase 3 Action</span>
      <span>{_badge(gt.action_type, sc_color)}</span>
    </div>
    <div class='meta-item'>
      <span class='meta-label'>Target Team</span>
      <span>{team_html}</span>
    </div>
    <div class='meta-item'>
      <span class='meta-label'>Response Keywords</span>
      <span>{kw_html}</span>
    </div>
  </div>
</div>"""
    return out


def _try_it_html() -> str:
    model = _esc(MODEL_NAME)
    return f"""
<h2 class='section-h'>Run the baseline yourself</h2>
<p class='section-sub'>The validator runs <code>inference.py</code> and parses structured stdout logs. Reproduce it locally with:</p>
<div class='code-block'>
<pre># Set environment variables
export API_BASE_URL=https://your-llm-proxy/v1
export HF_TOKEN=your_token_here
export MODEL_NAME={model}

# Run all 3 tasks (easy / medium / hard)
python inference.py</pre>
</div>
<h3 class='section-h' style='margin-top:32px'>Expected output format</h3>
<div class='code-block'>
<pre>[INFO] Using API_BASE_URL=https://... MODEL_NAME={model}

[START] task=easy env=support_env model={model}
[STEP]  step=1 action={{"action_type":"classify"}}                  reward=1.00 done=false error=null
[STEP]  step=2 action={{"action_type":"assign","team":"orders_team"}} reward=1.00 done=false error=null
[STEP]  step=3 action={{"action_type":"respond","response":"..."}}   reward=0.87 done=true  error=null
[END]   success=true steps=3 score=0.957 rewards=1.00,1.00,0.87

[START] task=medium env=support_env model={model}
...
[START] task=hard env=support_env model={model}
...</pre>
</div>
<h3 class='section-h' style='margin-top:32px'>Reward scoring summary</h3>
<div class='scoring-grid'>
  <div class='scoring-row header'><span>Phase</span><span>Correct</span><span>Wrong</span></div>
  <div class='scoring-row'><span style='color:#60a5fa'>Phase 1 — Triage</span><span style='color:#00d084'>0.998</span><span style='color:#ff6b6b'>0.002</span></div>
  <div class='scoring-row'><span style='color:#ffa94d'>Phase 2 — Route</span><span style='color:#00d084'>0.998</span><span style='color:#ffa94d'>0.400 (wrong team)</span></div>
  <div class='scoring-row'><span style='color:#00d084'>Phase 3 — Resolve</span><span style='color:#00d084'>0.002 – 0.998</span><span style='color:#ff6b6b'>penalties apply</span></div>
</div>
<p class='muted' style='margin-top:16px;font-size:12px'>All scores clamped to [0.002, 0.998]. Episode score = mean of all step rewards.</p>"""

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

.gradio-container {
  background: #0a0a0a !important;
  color: #ededed !important;
  font-family: Inter, system-ui, sans-serif !important;
  max-width: 100% !important;
  padding: 0 28px 32px !important;
}

/* ── Tabs ─────────────────────────────────────────────────────────── */
.tab-nav button {
  background: transparent !important;
  color: #9ca3af !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  font-family: Inter, sans-serif !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  padding: 10px 18px !important;
  transition: color .15s, border-color .15s !important;
}
.tab-nav button:hover { color: #ededed !important; }
.tab-nav button.selected {
  color: #00d084 !important;
  border-bottom-color: #00d084 !important;
}
.tabitem { background: transparent !important; border: none !important; padding: 24px 0 0 !important; }

/* ── Hero bar ─────────────────────────────────────────────────────── */
.hero {
  display: flex;
  gap: 32px;
  align-items: center;
  padding: 28px 32px;
  background: #141414;
  border: 1px solid #262626;
  border-radius: 10px;
  margin-bottom: 28px;
  flex-wrap: wrap;
}
.hero-title { flex: 1; }
.hero-title h1 { font-size: 22px; font-weight: 700; color: #ededed; line-height: 1.3; }
.hero-title p  { font-size: 13px; color: #9ca3af; margin-top: 4px; }
.hero-stats { display: flex; gap: 28px; flex-wrap: wrap; }
.stat-pill  {
  display: flex; flex-direction: column; align-items: center;
  padding: 8px 18px;
  background: #0a0a0a;
  border: 1px solid #262626;
  border-radius: 8px;
  min-width: 80px;
}
.stat-pill .num { font-size: 22px; font-weight: 700; color: #00d084; font-family: JetBrains Mono, monospace; }
.stat-pill .lbl { font-size: 11px; color: #9ca3af; margin-top: 2px; text-transform: uppercase; letter-spacing: .5px; }

/* ── Section typography ───────────────────────────────────────────── */
.section-h   { font-size: 17px; font-weight: 600; color: #ededed; margin: 28px 0 8px; }
.section-sub { font-size: 13px; color: #9ca3af; margin-bottom: 20px; line-height: 1.6; }
.muted       { color: #9ca3af; }
.tier-header { font-size: 13px; font-weight: 600; letter-spacing: 1px; margin: 28px 0 12px; font-family: JetBrains Mono, monospace; }

/* ── Unique cards grid ────────────────────────────────────────────── */
.unique-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 16px;
}
.unique-card {
  background: #141414;
  border: 1px solid #262626;
  border-radius: 10px;
  padding: 20px;
  transition: border-color .2s, transform .2s;
}
.unique-card:hover { border-color: #00d084; transform: translateY(-2px); }
.card-icon  { font-size: 24px; margin-bottom: 10px; }
.card-title { font-size: 14px; font-weight: 600; color: #ededed; margin-bottom: 6px; }
.card-body  { font-size: 13px; color: #9ca3af; line-height: 1.6; }

/* ── Architecture box ─────────────────────────────────────────────── */
.arch-box { background: #141414; border: 1px solid #262626; border-radius: 10px; overflow: hidden; }

/* ── Scenario cards ───────────────────────────────────────────────── */
.scen-card {
  background: #141414;
  border: 1px solid #262626;
  border-radius: 10px;
  padding: 18px 20px;
  margin-bottom: 12px;
  transition: border-color .2s;
}
.scen-card:hover { border-color: #404040; }
.scen-header { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; }
.ticket-id   { font-family: JetBrains Mono, monospace; font-size: 12px; color: #9ca3af; margin-left: auto; }
.scen-message {
  font-size: 13px; color: #ededed; line-height: 1.6;
  border-left: 3px solid #00d08444;
  padding: 8px 14px;
  background: #0a0a0a;
  border-radius: 0 6px 6px 0;
  margin-bottom: 14px;
  font-style: italic;
}
.scen-meta   { display: flex; gap: 20px; flex-wrap: wrap; }
.meta-item   { display: flex; flex-direction: column; gap: 4px; }
.meta-label  { font-size: 11px; color: #9ca3af; text-transform: uppercase; letter-spacing: .5px; font-weight: 600; }
.kw-tag      {
  display: inline-block; padding: 1px 7px; border-radius: 4px;
  background: #1a1a1a; color: #9ca3af; border: 1px solid #333;
  font-size: 11px; font-family: JetBrains Mono, monospace; margin: 2px;
}

/* ── Playground ───────────────────────────────────────────────────── */
.pg-placeholder { color: #5a5a5a; font-size: 13px; padding: 16px; font-style: italic; }
.pg-code {
  background: #0a0a0a;
  border: 1px solid #262626;
  border-radius: 8px;
  padding: 16px;
  font-family: JetBrains Mono, monospace;
  font-size: 12px;
  color: #9ca3af;
  overflow-x: auto;
  white-space: pre;
  line-height: 1.6;
}
.pg-status {
  display: flex;
  gap: 0;
  background: #141414;
  border: 1px solid #262626;
  border-radius: 8px;
  overflow: hidden;
}
.stat-item {
  display: flex; flex-direction: column; gap: 4px;
  padding: 14px 20px;
  border-right: 1px solid #262626;
  flex: 1;
}
.stat-item:last-child { border-right: none; }
.stat-label { font-size: 11px; color: #9ca3af; text-transform: uppercase; letter-spacing: .5px; font-weight: 600; }
.stat-val   { font-size: 15px; font-weight: 700; color: #ededed; font-family: JetBrains Mono, monospace; }
.score-badge {
  display: inline-block; padding: 2px 9px; border-radius: 4px;
  font-size: 12px; font-weight: 600; font-family: JetBrains Mono, monospace;
  margin: 2px;
}

/* ── Trace steps ─────────────────────────────────────────────────── */
.trace-wrap { display: flex; flex-direction: column; gap: 6px; }
.trace-step {
  padding: 10px 14px;
  border-radius: 6px;
  font-size: 12px;
  line-height: 1.5;
}
.trace-step code { font-family: JetBrains Mono, monospace; color: #ededed; background: none; }

/* ── Team assignment ─────────────────────────────────────────────── */
.team-wrap { display: flex; flex-direction: column; gap: 6px; }
.team-row  {
  padding: 10px 14px;
  background: #141414;
  border: 1px solid #ffa94d33;
  border-left: 3px solid #ffa94d;
  border-radius: 6px;
  font-size: 12px;
}
.team-row code { font-family: JetBrains Mono, monospace; color: #ffa94d; background: none; }

/* ── History ─────────────────────────────────────────────────────── */
.hist-wrap { display: flex; flex-direction: column; gap: 4px; }
.hist-row  { padding: 7px 12px; border-radius: 5px; font-size: 12px; line-height: 1.5; }
.hist-agent  { background: #60a5fa0d; border-left: 3px solid #60a5fa; color: #ededed; }
.hist-system { background: #0a0a0a; border-left: 3px solid #333; color: #9ca3af; font-style: italic; }

/* ── Agent message ───────────────────────────────────────────────── */
.agent-msg {
  background: #141414;
  border: 1px solid #00d08433;
  border-left: 3px solid #00d084;
  border-radius: 8px;
  padding: 14px 18px;
  font-size: 13px;
  line-height: 1.7;
  color: #ededed;
}

/* ── Try It code blocks ──────────────────────────────────────────── */
.code-block {
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 8px;
  overflow: hidden;
}
.code-block pre {
  padding: 18px 20px;
  font-family: JetBrains Mono, monospace;
  font-size: 12px;
  color: #9ca3af;
  line-height: 1.7;
  white-space: pre;
  overflow-x: auto;
}
.scoring-grid {
  background: #141414;
  border: 1px solid #262626;
  border-radius: 8px;
  overflow: hidden;
}
.scoring-row {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  padding: 10px 16px;
  border-bottom: 1px solid #1f1f1f;
  font-size: 13px;
  font-family: JetBrains Mono, monospace;
}
.scoring-row.header {
  background: #0a0a0a;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: .5px;
  color: #5a5a5a;
  font-weight: 600;
}
.scoring-row:last-child { border-bottom: none; }

/* ── Gradio component overrides ──────────────────────────────────── */
.gradio-container label { color: #9ca3af !important; font-size: 12px !important; }
.gradio-container input, .gradio-container select, .gradio-container textarea {
  background: #141414 !important;
  border-color: #262626 !important;
  color: #ededed !important;
  font-family: JetBrains Mono, monospace !important;
  font-size: 12px !important;
}
.gradio-container .gr-button-primary, button.primary {
  background: #00d084 !important;
  color: #0a0a0a !important;
  border: 1px solid #00a866 !important;
  font-weight: 600 !important;
  font-family: Inter, sans-serif !important;
}
.gradio-container .gr-button-primary:hover { background: #00b870 !important; }
.gradio-container .gr-button-secondary, button.secondary {
  background: #141414 !important;
  color: #ededed !important;
  border: 1px solid #262626 !important;
}
.gradio-container .gr-button-secondary:hover { border-color: #404040 !important; }
.gradio-container .gr-accordion {
  background: #141414 !important;
  border-color: #262626 !important;
}
.gradio-container .block.svelte-1gfkfd6 {
  background: transparent !important;
  border: none !important;
}

/* ── Footer ──────────────────────────────────────────────────────── */
.footer {
  margin-top: 48px;
  padding-top: 20px;
  border-top: 1px solid #1f1f1f;
  text-align: center;
  font-size: 12px;
  color: #5a5a5a;
  line-height: 1.8;
}
.footer a { color: #9ca3af; text-decoration: none; }
.footer a:hover { color: #00d084; }
"""

HERO_HTML = """
<div class='hero'>
  <div class='hero-title'>
    <h1>Customer Support Agent</h1>
    <p>OpenEnv-compliant autonomous BPO agent · Meta PyTorch Hackathon x Scaler</p>
  </div>
  <div class='hero-stats'>
    <div class='stat-pill'><span class='num'>15</span><span class='lbl'>Tickets</span></div>
    <div class='stat-pill'><span class='num'>3</span><span class='lbl'>Tiers</span></div>
    <div class='stat-pill'><span class='num'>3</span><span class='lbl'>Phases</span></div>
    <div class='stat-pill'><span class='num'>6</span><span class='lbl'>Teams</span></div>
    <div class='stat-pill'><span class='num'>5</span><span class='lbl'>Max Steps</span></div>
  </div>
</div>
"""

FOOTER_HTML = """
<div class='footer'>
  customer-support-agent · 3-phase MDP · 15 tickets · 6 specialist teams · deterministic grader · reward ∈ [0.002, 0.998]<br/>
  <a href='https://github.com/AdityaK-labs/Support-Agent/tree/Tarun'>github</a> ·
  <a href='https://huggingface.co/spaces/Tarun21W/MetaAI'>huggingface</a>
</div>
"""

# ---------------------------------------------------------------------------
# Gradio UI functions
# ---------------------------------------------------------------------------

def ui_reset(task_choice: str):
    global _agent_history
    _agent_history = []
    task_name = TASK_MAP.get(task_choice, "easy")
    res = env.reset(task_name)
    obs_json = res.observation.model_dump_json(indent=2)
    return (
        _obs_html(obs_json),
        _msg_html(""),
        _steps_html([]),
        "<div class='pg-placeholder'>No scores yet.</div>",
        _history_html(),
        _team_html([]),
    )


async def ui_auto_step():
    global _agent_history

    if env.done:
        return (
            _obs_html(env.get_current_observation().model_dump_json(indent=2)),
            _msg_html(""),
            "<div class='pg-placeholder'>Episode finished — please reset.</div>",
            "<div class='pg-placeholder'>Episode finished.</div>",
            _history_html(),
            _team_html([]),
        )

    all_steps  = []
    all_rewards = []
    last_message = ""
    team_lines  = []

    for step_num in range(1, 6):
        if env.done:
            break

        obs = env.get_current_observation()
        current_phase = env.phase

        try:
            action_data = await _call_llm(obs, _agent_history, phase=current_phase)
            action = Action(**action_data)
            error_msg = None
        except Exception as exc:
            error_msg = str(exc)
            if current_phase == 1:
                action = Action(action_type="classify")
            elif current_phase == 2:
                action = Action(action_type="assign", team="management_team")
            else:
                action = Action(action_type="respond", response="Thank you for contacting us.")

        result  = env.step(action)
        reward  = result.reward
        done    = result.done

        _agent_history = env.history.copy()
        all_rewards.append(reward)

        phase_label = _PHASE_NAMES.get(current_phase, str(current_phase))
        step_line = f"[Phase {current_phase}: {phase_label}]  step={step_num}  {action.action_type}"
        if action.team:     step_line += f" → {action.team}"
        step_line += f"  |  score: {reward:.3f}"
        if error_msg:       step_line += f"  ⚠ {error_msg[:60]}"
        all_steps.append(step_line)

        if action.team:
            desc = TEAM_DESCRIPTIONS.get(action.team, action.team)
            sub  = TEAM_SUBTITLES.get(action.team, "")
            tid  = obs.ticket_id if hasattr(obs, "ticket_id") else "?"
            team_lines.append(f"[{tid}]  {desc} — {sub}")

        if action.response:
            last_message = action.response

        if done:
            break

    avg = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    return (
        _obs_html(env.get_current_observation().model_dump_json(indent=2)),
        _msg_html(last_message),
        _steps_html(all_steps),
        _score_html(all_rewards, avg, env.done),
        _history_html(),
        _team_html(team_lines),
    )


def ui_manual_step(a_type: str, t_name: str, resp_text: str):
    global _agent_history

    if env.done:
        return (
            _obs_html(env.get_current_observation().model_dump_json(indent=2)),
            _msg_html(""),
            "<div class='pg-placeholder'>Episode finished — please reset.</div>",
            "<div class='pg-placeholder'>Episode finished.</div>",
            _history_html(),
            _team_html([]),
        )

    try:
        action = Action(
            action_type=a_type,
            team=t_name or None,
            response=resp_text or None,
        )
        result = env.step(action)
        reward = result.reward
        done   = result.done
        _agent_history = env.history.copy()

        phase_num = result.info.get("phase", env.phase)
        phase_lbl = _PHASE_NAMES.get(phase_num - 1, str(phase_num - 1))
        step_line = f"[Phase {phase_num - 1}: {phase_lbl}]  {a_type}"
        if t_name: step_line += f" → {t_name}"
        step_line += f"  |  score: {reward:.3f}"

        team_lines = []
        if t_name:
            desc = TEAM_DESCRIPTIONS.get(t_name, t_name)
            sub  = TEAM_SUBTITLES.get(t_name, "")
            team_lines.append(f"[{env.current_scenario.ticket.ticket_id}]  {desc} — {sub}")

        return (
            _obs_html(result.observation.model_dump_json(indent=2)),
            _msg_html(resp_text or ""),
            _steps_html([step_line]),
            _manual_score_html(reward, done, result.info.get("feedback", "")),
            _history_html(),
            _team_html(team_lines),
        )

    except Exception as exc:
        return (
            _obs_html(env.get_current_observation().model_dump_json(indent=2)),
            _msg_html(""),
            f"<div class='pg-placeholder' style='color:#ff6b6b'>Error: {_esc(str(exc))}</div>",
            "<div class='pg-placeholder'>—</div>",
            _history_html(),
            _team_html([]),
        )

# ---------------------------------------------------------------------------
# Gradio layout
# ---------------------------------------------------------------------------

with gr.Blocks(css=CSS, title="Customer Support Agent — OpenEnv") as demo:

    gr.HTML(HERO_HTML)

    with gr.Tabs():

        # ── Tab 1: Overview ───────────────────────────────────────────
        with gr.Tab("Overview"):
            gr.HTML(_overview_html())

        # ── Tab 2: Scenarios ──────────────────────────────────────────
        with gr.Tab("Scenarios"):
            gr.HTML(f"<h2 class='section-h'>15 support tickets · 3 difficulty tiers</h2>")
            gr.HTML(f"<p class='section-sub'>Each scenario runs the full 3-phase MDP. Phase 3 action and keywords shown are the ground truth the grader checks against.</p>")
            gr.HTML(_scenarios_html())

        # ── Tab 3: Playground ─────────────────────────────────────────
        with gr.Tab("Playground"):
            gr.HTML("<h2 class='section-h'>Live interactive episode</h2>")
            gr.HTML("<p class='section-sub'>Select a task tier, reset the environment, then run the full 3-phase episode automatically — or step through each phase manually.</p>")

            with gr.Row():
                task_dd   = gr.Dropdown(choices=list(TASK_MAP.keys()), value="Easy (Classification)", label="Task Tier", scale=4)
                reset_btn = gr.Button("Reset Episode", variant="secondary", scale=1, min_width=140)
                auto_btn  = gr.Button("Run Agent (Full Episode)", variant="primary", scale=2, min_width=200)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("<div class='section-h' style='margin:16px 0 8px'>Current Observation</div>")
                    obs_out = gr.HTML(_obs_html(""), label="")

                    gr.HTML("<div class='section-h' style='margin:16px 0 8px'>Action History</div>")
                    hist_out = gr.HTML(_history_html(), label="")

                with gr.Column(scale=1):
                    gr.HTML("<div class='section-h' style='margin:16px 0 8px'>Phase Steps</div>")
                    steps_out = gr.HTML(_steps_html([]), label="")

                    gr.HTML("<div class='section-h' style='margin:16px 0 8px'>Team Assignment</div>")
                    team_out = gr.HTML(_team_html([]), label="")

                    gr.HTML("<div class='section-h' style='margin:16px 0 8px'>Agent Response</div>")
                    msg_out = gr.HTML(_msg_html(""), label="")

                    gr.HTML("<div class='section-h' style='margin:16px 0 8px'>Score</div>")
                    score_out = gr.HTML("<div class='pg-placeholder'>No scores yet.</div>", label="")

            with gr.Accordion("Manual Step (override agent)", open=False):
                with gr.Row():
                    act_type = gr.Dropdown(choices=["classify","assign","respond","refund","escalate"], value="classify", label="Action Type", scale=1)
                    act_team = gr.Textbox(label="Team (optional)", scale=1)
                with gr.Row():
                    act_resp   = gr.Textbox(label="Response text (optional)", lines=2, scale=3)
                    manual_btn = gr.Button("Submit Manual Action", variant="primary", scale=1, min_width=160)

        # ── Tab 4: Try It ─────────────────────────────────────────────
        with gr.Tab("Try It"):
            gr.HTML(_try_it_html())

    gr.HTML(FOOTER_HTML)

    # ── Event wiring ─────────────────────────────────────────────────
    _outputs = [obs_out, msg_out, steps_out, score_out, hist_out, team_out]

    reset_btn.click(ui_reset,        inputs=[task_dd],                           outputs=_outputs)
    auto_btn.click( ui_auto_step,    inputs=[],                                  outputs=_outputs)
    manual_btn.click(ui_manual_step, inputs=[act_type, act_team, act_resp],      outputs=_outputs)

demo.queue()
app = gr.mount_gradio_app(app, demo, path="/")

def main():
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=True)

if __name__ == "__main__":
    main()