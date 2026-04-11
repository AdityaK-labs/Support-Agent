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
# Design tokens
# ---------------------------------------------------------------------------

_PHASE_COLORS  = {1: "#60a5fa", 2: "#ffa94d", 3: "#00d084"}
_PHASE_NAMES   = {1: "Triage",  2: "Route",    3: "Resolve"}
_TIER_COLORS   = {"easy": "#00d084", "medium": "#ffa94d", "hard": "#ff6b6b"}
_SENT_COLORS   = {"positive": "#00d084", "neutral": "#9ca3af", "negative": "#ffa94d", "angry": "#ff6b6b"}
_PRIO_COLORS   = {"low": "#9ca3af", "medium": "#60a5fa", "high": "#ffa94d", "critical": "#ff6b6b"}
_ACTION_COLORS = {"classify": "#60a5fa", "assign": "#ffa94d", "respond": "#00d084", "refund": "#c084fc", "escalate": "#ff6b6b"}
_TEAM_NAMES    = {
    "logistics_team":    "Logistics Team",
    "tech_support_team": "Tech Support Team",
    "safety_team":       "Safety Team",
    "finance_team":      "Finance Team",
    "orders_team":       "Orders Team",
    "management_team":   "Management Team",
}
_TEAM_SUBTITLES = {
    "logistics_team":    "Shipping, delivery, lost packages",
    "tech_support_team": "Login, bugs, API errors, data loss",
    "safety_team":       "Defects, overheating, recalls, hazards",
    "finance_team":      "Invoices, billing, tax, payments",
    "orders_team":       "Bulk orders, returns, subscriptions",
    "management_team":   "Escalated complaints, manager requests",
}

_agent_history: List[str] = []

# ---------------------------------------------------------------------------
# HTML utility helpers
# ---------------------------------------------------------------------------

def _esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def _score_color(s: float) -> str:
    if s >= 0.80: return "#00d084"
    if s >= 0.50: return "#ffa94d"
    return "#ff6b6b"

def _badge(label: str, color: str, size: str = "12") -> str:
    return (
        f"<span style='display:inline-flex;align-items:center;padding:2px 9px;"
        f"border-radius:4px;background:{color}1a;color:{color};"
        f"border:1px solid {color}44;font-size:{size}px;font-weight:600;"
        f"font-family:\"JetBrains Mono\",monospace;letter-spacing:.3px;"
        f"text-transform:uppercase;line-height:1.6'>{_esc(label)}</span>"
    )

def _icon_box(svg_path: str) -> str:
    return (
        f"<div style='width:40px;height:40px;background:#0a2a1a;"
        f"border:1px solid #1a4a2a;border-radius:8px;"
        f"display:flex;align-items:center;justify-content:center;margin-bottom:14px;flex-shrink:0'>"
        f"<svg width='18' height='18' viewBox='0 0 24 24' fill='none' "
        f"stroke='#00d084' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>"
        f"{svg_path}</svg></div>"
    )

# SVG icon paths (Lucide-style)
_ICONS = {
    "zap":    "<polygon points='13 2 3 14 12 14 11 22 21 10 12 10 13 2'/>",
    "layers": "<polygon points='12 2 2 7 12 12 22 7 12 2'/><polyline points='2 17 12 22 22 17'/><polyline points='2 12 12 17 22 12'/>",
    "git":    "<line x1='6' y1='3' x2='6' y2='15'/><circle cx='18' cy='6' r='3'/><circle cx='6' cy='18' r='3'/><path d='M18 9a9 9 0 0 1-9 9'/>",
    "scale":  "<line x1='12' y1='3' x2='12' y2='21'/><path d='M3 9l4.5 9'/><path d='M21 9l-4.5 9'/><path d='M3 18a4.5 4.5 0 0 0 9 0'/><path d='M12 18a4.5 4.5 0 0 0 9 0'/>",
    "bar":    "<line x1='18' y1='20' x2='18' y2='10'/><line x1='12' y1='20' x2='12' y2='4'/><line x1='6' y1='20' x2='6' y2='14'/><line x1='2' y1='20' x2='22' y2='20'/>",
    "lock":   "<rect x='3' y='11' width='18' height='11' rx='2' ry='2'/><path d='M7 11V7a5 5 0 0 1 10 0v4'/>",
}

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Global reset ─────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

.gradio-container {
  background: #0d0d0d !important;
  color: #e8e8e8 !important;
  font-family: Inter, system-ui, -apple-system, sans-serif !important;
  max-width: 100% !important;
  padding: 0 !important;
  min-height: 100vh !important;
}
.gradio-container > .main { padding: 0 32px 40px !important; }

/* ── Strip default Gradio chrome from HTML containers ────────────── */
.gradio-container .block,
.gradio-container .form,
.gradio-container .gap {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
  gap: 0 !important;
}
.gradio-container .wrap { border: none !important; }
.gradio-container .output-html { padding: 0 !important; }

/* ── Tab bar ──────────────────────────────────────────────────────── */
.gradio-container .tab-nav {
  background: #0d0d0d !important;
  border-bottom: 1px solid #1f1f1f !important;
  padding: 0 32px !important;
  gap: 0 !important;
}
.gradio-container .tab-nav button {
  background: transparent !important;
  color: #666 !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  font-family: Inter, sans-serif !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  padding: 14px 20px !important;
  margin: 0 !important;
  border-radius: 0 !important;
  transition: color .15s, border-color .15s !important;
  letter-spacing: .2px !important;
}
.gradio-container .tab-nav button:hover { color: #ccc !important; }
.gradio-container .tab-nav button.selected {
  color: #00d084 !important;
  border-bottom-color: #00d084 !important;
}
.gradio-container .tabitem {
  background: transparent !important;
  border: none !important;
  padding: 32px 0 0 !important;
}

/* ── Gradio input component overrides ────────────────────────────── */
.gradio-container select,
.gradio-container input[type=text],
.gradio-container textarea {
  background: #1a1a1a !important;
  border: 1px solid #2a2a2a !important;
  color: #e8e8e8 !important;
  font-family: "JetBrains Mono", monospace !important;
  font-size: 12px !important;
  border-radius: 6px !important;
}
.gradio-container select:focus,
.gradio-container input:focus,
.gradio-container textarea:focus {
  border-color: #00d084 !important;
  outline: none !important;
  box-shadow: 0 0 0 2px #00d08422 !important;
}
.gradio-container label span {
  color: #666 !important;
  font-size: 11px !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: .6px !important;
  font-family: Inter, sans-serif !important;
}
button.primary, .gradio-container button.primary {
  background: #00d084 !important;
  color: #0a0a0a !important;
  border: 1px solid #00a866 !important;
  border-radius: 6px !important;
  font-family: Inter, sans-serif !important;
  font-weight: 600 !important;
  font-size: 13px !important;
  transition: background .15s !important;
}
button.primary:hover { background: #00b870 !important; }
button.secondary, .gradio-container button.secondary {
  background: #1a1a1a !important;
  color: #ccc !important;
  border: 1px solid #2a2a2a !important;
  border-radius: 6px !important;
  font-family: Inter, sans-serif !important;
  font-weight: 500 !important;
  font-size: 13px !important;
  transition: border-color .15s !important;
}
button.secondary:hover { border-color: #444 !important; color: #e8e8e8 !important; }
.gradio-container .gr-accordion {
  background: #141414 !important;
  border: 1px solid #222 !important;
  border-radius: 8px !important;
  margin-top: 16px !important;
}
.gradio-container .gr-accordion > .label-wrap {
  padding: 12px 16px !important;
  color: #888 !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: .5px !important;
}

/* ── Hero ─────────────────────────────────────────────────────────── */
.hero {
  background: #141414;
  border-bottom: 1px solid #1f1f1f;
  padding: 32px 32px 28px;
}
.hero-title {
  font-size: 26px;
  font-weight: 700;
  color: #f0f0f0;
  line-height: 1.2;
  margin-bottom: 10px;
  letter-spacing: -.3px;
}
.hero-desc {
  font-size: 13px;
  color: #666;
  line-height: 1.75;
  max-width: 860px;
}
.stat-grid {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 10px;
  margin-top: 24px;
}
.stat-card {
  background: #0d0d0d;
  border: 1px solid #1f1f1f;
  border-radius: 8px;
  padding: 18px 16px 14px;
}
.sc-label {
  font-size: 10px;
  font-weight: 600;
  color: #444;
  text-transform: uppercase;
  letter-spacing: 1.2px;
  margin-bottom: 10px;
}
.sc-num {
  font-size: 38px;
  font-weight: 700;
  color: #e8e8e8;
  font-family: "JetBrains Mono", monospace;
  line-height: 1;
}
.sc-num.sm { font-size: 22px; padding-top: 8px; }

/* ── Section typography ───────────────────────────────────────────── */
.section-h   { font-size: 18px; font-weight: 600; color: #f0f0f0; margin: 0 0 8px; }
.section-sub { font-size: 13px; color: #666; margin-bottom: 28px; line-height: 1.6; }

/* ── Overview unique cards ────────────────────────────────────────── */
.unique-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 14px;
  margin-top: 4px;
}
.unique-card {
  background: #141414;
  border: 1px solid #222;
  border-radius: 10px;
  padding: 22px 22px 20px;
  transition: border-color .2s, transform .2s;
}
.unique-card:hover { border-color: #00d08466; transform: translateY(-2px); }
.uc-title { font-size: 14px; font-weight: 600; color: #f0f0f0; margin-bottom: 8px; }
.uc-body  { font-size: 13px; color: #777; line-height: 1.7; }

/* ── Data tables (shared) ─────────────────────────────────────────── */
.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
  font-family: "JetBrains Mono", monospace;
}
.data-table thead tr {
  background: #0d0d0d;
  border-bottom: 2px solid #1f1f1f;
}
.data-table th {
  padding: 10px 14px;
  text-align: left;
  font-size: 10px;
  font-weight: 600;
  color: #444;
  text-transform: uppercase;
  letter-spacing: 1px;
  white-space: nowrap;
}
.data-table th.tc { text-align: center; }
.data-table th.tr { text-align: right; }
.data-table td {
  padding: 11px 14px;
  border-bottom: 1px solid #1a1a1a;
  vertical-align: middle;
  color: #ccc;
  line-height: 1.5;
}
.data-table td.tc { text-align: center; }
.data-table td.tr { text-align: right; }
.data-table tbody tr:hover { background: #161616; }
.data-table tbody tr:last-child td { border-bottom: none; }
.msg-cell {
  max-width: 320px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: #888;
  font-style: italic;
}
.kw-cell { max-width: 240px; }
.kw-tag {
  display: inline-block;
  padding: 1px 6px;
  border-radius: 3px;
  background: #1f1f1f;
  border: 1px solid #2a2a2a;
  color: #777;
  font-size: 10px;
  margin: 1px;
}

/* ── Tier section ─────────────────────────────────────────────────── */
.tier-section { margin-bottom: 32px; }
.tier-header-row {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 14px 16px;
  background: #111;
  border: 1px solid #1f1f1f;
  border-radius: 8px 8px 0 0;
  border-bottom: none;
}
.tier-info { font-size: 12px; color: #555; font-family: "JetBrains Mono", monospace; }
.table-wrap {
  background: #141414;
  border: 1px solid #1f1f1f;
  border-radius: 0 0 8px 8px;
  overflow: hidden;
}

/* ── Playground panels ────────────────────────────────────────────── */
.panel-title {
  font-size: 11px;
  font-weight: 600;
  color: #444;
  text-transform: uppercase;
  letter-spacing: 1px;
  padding: 0 0 8px;
  margin-bottom: 0;
}
.pg-panel {
  background: #141414;
  border: 1px solid #1f1f1f;
  border-radius: 8px;
  overflow: hidden;
}
.pg-panel-head {
  background: #111;
  border-bottom: 1px solid #1f1f1f;
  padding: 10px 16px;
  font-size: 11px;
  font-weight: 600;
  color: #555;
  text-transform: uppercase;
  letter-spacing: 1px;
}
.pg-panel-body { padding: 0; }
.empty-state {
  padding: 24px 16px;
  color: #3a3a3a;
  font-size: 13px;
  font-style: italic;
  text-align: center;
}

/* Observation table inside panel */
.obs-table { width: 100%; border-collapse: collapse; }
.obs-table td { padding: 9px 16px; border-bottom: 1px solid #1a1a1a; font-size: 12px; vertical-align: top; }
.obs-table tr:last-child td { border-bottom: none; }
.obs-key {
  width: 100px;
  color: #444;
  font-family: "JetBrains Mono", monospace;
  font-size: 11px;
  white-space: nowrap;
}
.obs-val { color: #ccc; font-family: "JetBrains Mono", monospace; font-size: 12px; word-break: break-word; }
.obs-message { font-family: Inter, sans-serif; font-size: 12px; color: #999; font-style: italic; line-height: 1.6; }
.hist-mini { font-size: 11px; color: #555; line-height: 1.6; }

/* Episode score metrics bar */
.metrics-bar {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  border: 1px solid #1f1f1f;
  border-radius: 8px;
  overflow: hidden;
  background: #141414;
}
.metric-cell {
  padding: 16px;
  border-right: 1px solid #1f1f1f;
}
.metric-cell:last-child { border-right: none; }
.metric-label { font-size: 10px; font-weight: 600; color: #444; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
.metric-val   { font-size: 28px; font-weight: 700; color: #e8e8e8; font-family: "JetBrains Mono", monospace; line-height: 1; }
.metric-val.sm { font-size: 14px; padding-top: 7px; }
.score-pill {
  display: inline-flex; align-items: center;
  padding: 3px 10px; border-radius: 4px;
  font-size: 12px; font-weight: 700;
  font-family: "JetBrains Mono", monospace;
  margin: 2px;
}

/* Team assignment card */
.team-card {
  display: flex;
  align-items: flex-start;
  gap: 14px;
  padding: 16px;
  background: #141414;
  border: 1px solid #1f1f1f;
  border-left: 3px solid #ffa94d;
  border-radius: 8px;
  margin-bottom: 8px;
}
.team-icon {
  width: 36px; height: 36px;
  background: #1f1500;
  border: 1px solid #3a2800;
  border-radius: 6px;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
  font-size: 16px;
}
.team-name { font-size: 13px; font-weight: 600; color: #e8e8e8; margin-bottom: 3px; }
.team-sub  { font-size: 11px; color: #666; }

/* Agent response */
.agent-response {
  padding: 16px;
  background: #141414;
  border: 1px solid #00d08422;
  border-left: 3px solid #00d084;
  border-radius: 8px;
  font-size: 13px;
  color: #ccc;
  line-height: 1.75;
  font-family: Inter, sans-serif;
}

/* Try It code blocks */
.code-block {
  background: #0d1117;
  border: 1px solid #21262d;
  border-radius: 8px;
  overflow: hidden;
  margin: 16px 0;
}
.code-block pre {
  padding: 20px 24px;
  font-family: "JetBrains Mono", monospace;
  font-size: 12px;
  color: #8b949e;
  line-height: 1.75;
  white-space: pre;
  overflow-x: auto;
  margin: 0;
}

/* Footer */
.footer {
  margin-top: 48px;
  padding: 20px 32px;
  border-top: 1px solid #1a1a1a;
  font-size: 12px;
  color: #3a3a3a;
  line-height: 1.9;
}
.footer a { color: #555; text-decoration: none; }
.footer a:hover { color: #00d084; }
"""

# ---------------------------------------------------------------------------
# Static HTML sections
# ---------------------------------------------------------------------------

def _hero_html() -> str:
    desc = (
        "A 3-phase Markov Decision Process where LLM agents triage, route, and resolve "
        "customer support tickets across a structured action space — classify, assign, respond, "
        "refund, escalate. Covers e-commerce and SaaS BPO workflows across 3 difficulty tiers. "
        "Deterministic per-phase reward shaping. Parameter-randomized ticket selection prevents "
        "memorization — agents must learn the decision pattern, not specific tickets."
    )
    return f"""
<div class='hero'>
  <div class='hero-title'>Customer Support Agent</div>
  <div class='hero-desc'>{desc}</div>
  <div class='stat-grid'>
    <div class='stat-card'><div class='sc-label'>Tickets</div><div class='sc-num'>15</div></div>
    <div class='stat-card'><div class='sc-label'>Tiers</div><div class='sc-num'>3</div></div>
    <div class='stat-card'><div class='sc-label'>Phases</div><div class='sc-num'>3</div></div>
    <div class='stat-card'><div class='sc-label'>Teams</div><div class='sc-num'>6</div></div>
    <div class='stat-card'><div class='sc-label'>Max Steps</div><div class='sc-num'>5</div></div>
    <div class='stat-card'><div class='sc-label'>Reward</div><div class='sc-num sm'>[0.002, 0.998]</div></div>
  </div>
</div>"""


def _overview_html() -> str:
    cards = [
        ("zap",    "3-Phase MDP, Not a Bandit",
         "Every ticket runs Triage → Route → Resolve with state evolving between steps. "
         "The agent makes 3 sequential decisions per episode — a proper Markov Decision "
         "Process, not a single-shot classification task."),
        ("layers", "Dense Per-Phase Rewards",
         "Intermediate reward at every phase. No sparse end-of-episode signal. "
         "Phase 1 binary (classify or not), Phase 2 graded by team accuracy, "
         "Phase 3 proportional quality scoring across action + team + keyword coverage."),
        ("git",    "Evolving Observation State",
         "After Phase 1, issue_type updates from 'unknown' to the true category. "
         "After Phase 2, the assigned team appears in history. The agent sees richer "
         "context at each step — state transitions are real, not simulated."),
        ("scale",  "Asymmetric Business Penalties",
         "Refund (−0.50) > Escalation (−0.30) > Wrong Team (−0.15). "
         "Mirrors actual BPO cost: an unwarranted refund is immediate financial loss, "
         "a false escalation wastes senior time, wrong routing is recoverable."),
        ("bar",    "Proportional Normalization",
         "Phase 3 scores normalize raw_score / max_possible. A simple ticket needing "
         "only classify can still achieve 0.998 without being penalized for omitting "
         "team or response components it never required."),
        ("lock",   "Deterministic Grader",
         "Same action + same ticket = same reward. No stochasticity in the grader. "
         "Reproducible RL training and evaluation without environment noise "
         "contaminating the reward signal across runs."),
    ]
    grid = "".join(
        f"<div class='unique-card'>{_icon_box(_ICONS[icon])}"
        f"<div class='uc-title'>{title}</div><div class='uc-body'>{body}</div></div>"
        for icon, title, body in cards
    )
    arch = """<div class='code-block'><pre>Episode Flow — 3 sequential steps per ticket:

  reset()  →  Observation { issue_type: "unknown", sentiment, priority, message }

  step({"action_type": "classify"})               ←  Phase 1: Triage
     reward: 0.998  (correct) | 0.002 (wrong)
     effect: issue_type revealed in next observation

  step({"action_type": "assign", "team": "…"})    ←  Phase 2: Route
     reward: 0.998  (correct team) | 0.400 (wrong team) | 0.002 (wrong action)
     effect: team context appended to history

  step({"action_type": "respond|refund|escalate"}) ←  Phase 3: Resolve
     reward: proportional quality score in [0.002, 0.998]
     effect: done = True

  episode_score = mean(r1, r2, r3)</pre></div>"""
    return f"""
<h2 class='section-h'>What makes this unique</h2>
<p class='section-sub'>Six design decisions that differentiate this environment from single-step LLM evaluation benchmarks.</p>
<div class='unique-grid'>{grid}</div>
<h2 class='section-h' style='margin-top:40px'>Episode Architecture</h2>
<p class='section-sub'>Each episode is a deterministic 3-step MDP. Observation state updates between phases.</p>
{arch}"""


def _scenarios_html() -> str:
    out = ""
    tier_meta = {
        "easy":   ("5 tickets", "respond, refund"),
        "medium": ("5 tickets", "respond, escalate"),
        "hard":   ("5 tickets", "escalate, refund, respond"),
    }
    for tier in ["easy", "medium", "hard"]:
        tc = _TIER_COLORS[tier]
        tickets, actions = tier_meta[tier]
        badge = _badge(tier, tc)
        rows = ""
        for sc in TASK_REGISTRY[tier]:
            t  = sc.ticket
            gt = sc.ground_truth
            sc_color  = _ACTION_COLORS.get(gt.action_type, "#9ca3af")
            sent_c    = _SENT_COLORS.get(t.sentiment, "#9ca3af")
            prio_c    = _PRIO_COLORS.get(t.priority, "#9ca3af")
            kws = " ".join(
                f"<span class='kw-tag'>{_esc(k)}</span>"
                for k in gt.response_keywords
            ) if gt.response_keywords else "<span style='color:#333'>—</span>"
            team_html = (
                f"<code style='color:#ffa94d;font-size:11px'>{_esc(gt.team)}</code>"
                if gt.team else "<span style='color:#333'>—</span>"
            )
            msg_short = t.message[:72] + "…" if len(t.message) > 72 else t.message
            rows += f"""
<tr>
  <td><code style='color:#666;font-size:11px'>{_esc(t.ticket_id)}</code></td>
  <td><span style='color:{sent_c};font-weight:500'>{_esc(t.sentiment)}</span></td>
  <td><span style='color:{prio_c};font-weight:500'>{_esc(t.priority)}</span></td>
  <td class='msg-cell' title='{_esc(t.message)}'>&ldquo;{_esc(msg_short)}&rdquo;</td>
  <td>{_badge(gt.action_type, sc_color, "11")}</td>
  <td>{team_html}</td>
  <td class='kw-cell'>{kws}</td>
</tr>"""
        out += f"""
<div class='tier-section'>
  <div class='tier-header-row'>
    {badge}
    <span class='tier-info'>{tickets} · Phase 3 actions: {actions}</span>
  </div>
  <div class='table-wrap'>
    <table class='data-table'>
      <thead>
        <tr>
          <th>Ticket ID</th>
          <th>Sentiment</th>
          <th>Priority</th>
          <th>Customer Message</th>
          <th>Phase 3 Action</th>
          <th>Target Team</th>
          <th>Response Keywords</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</div>"""
    return f"""
<h2 class='section-h'>15 support tickets · 3 difficulty tiers</h2>
<p class='section-sub'>
  Each scenario runs the full 3-phase MDP. Hover over a message cell to read the full text.
  Phase 3 action and keywords are ground truth — the deterministic grader checks against these.
</p>
{out}"""


def _try_it_html() -> str:
    m = _esc(MODEL_NAME)
    return f"""
<h2 class='section-h'>Run the baseline yourself</h2>
<p class='section-sub'>
  The validator runs <code style='color:#9ca3af;font-size:12px'>inference.py</code> and parses structured stdout logs.
  Set the two required env vars and run:
</p>
<div class='code-block'><pre>export API_BASE_URL=https://your-llm-proxy/v1
export HF_TOKEN=your_token_here
export MODEL_NAME={m}        # optional

python inference.py</pre></div>

<h2 class='section-h' style='margin-top:36px'>Expected output</h2>
<div class='code-block'><pre>[INFO] Using API_BASE_URL=https://... MODEL_NAME={m}

[START] task=easy   env=support_env model={m}
[STEP]  step=1  action={{"action_type":"classify"}}                     reward=1.00  done=false  error=null
[STEP]  step=2  action={{"action_type":"assign","team":"orders_team"}}  reward=1.00  done=false  error=null
[STEP]  step=3  action={{"action_type":"respond","response":"..."}}     reward=0.87  done=true   error=null
[END]   success=true  steps=3  score=0.957  rewards=1.00,1.00,0.87

[START] task=medium  ...
[START] task=hard    ...</pre></div>

<h2 class='section-h' style='margin-top:36px'>Reward breakdown</h2>
<div class='table-wrap' style='border-radius:8px'>
<table class='data-table'>
  <thead>
    <tr>
      <th>Phase</th>
      <th>Correct</th>
      <th>Wrong action type</th>
      <th>Wrong team</th>
      <th>Penalty</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><span style='color:#60a5fa;font-weight:600'>Phase 1 — Triage</span></td>
      <td><span style='color:#00d084;font-weight:700'>0.998</span></td>
      <td><span style='color:#ff6b6b'>0.002</span></td>
      <td><span style='color:#555'>n/a</span></td>
      <td><span style='color:#555'>—</span></td>
    </tr>
    <tr>
      <td><span style='color:#ffa94d;font-weight:600'>Phase 2 — Route</span></td>
      <td><span style='color:#00d084;font-weight:700'>0.998</span></td>
      <td><span style='color:#ff6b6b'>0.002</span></td>
      <td><span style='color:#ffa94d'>0.400</span></td>
      <td><span style='color:#555'>—</span></td>
    </tr>
    <tr>
      <td><span style='color:#00d084;font-weight:600'>Phase 3 — Resolve</span></td>
      <td><span style='color:#00d084;font-weight:700'>0.002 – 0.998</span></td>
      <td><span style='color:#ff6b6b'>0.002</span></td>
      <td><span style='color:#ff6b6b'>−0.15</span></td>
      <td><span style='color:#ff6b6b'>refund −0.50 · escalate −0.30</span></td>
    </tr>
    <tr style='background:#0d0d0d;font-weight:600'>
      <td style='color:#888'>Episode Score</td>
      <td colspan='4' style='color:#888'>mean(phase1, phase2, phase3) · clamped to [0.002, 0.998]</td>
    </tr>
  </tbody>
</table>
</div>"""

# ---------------------------------------------------------------------------
# Playground HTML renderers
# ---------------------------------------------------------------------------

def _obs_html(obs_json: str) -> str:
    if not obs_json:
        return "<div class='empty-state'>Reset the environment to load a ticket.</div>"
    try:
        obs = json.loads(obs_json)
    except Exception:
        return f"<div class='empty-state'>—</div>"

    def _val(key, val):
        if key == "ticket_id":
            return f"<code style='color:#888'>{_esc(val)}</code>"
        if key == "issue_type":
            if val == "unknown":
                return f"<span style='color:#333;font-style:italic'>unknown</span>"
            return f"<span style='color:#00d084;font-weight:600'>{_esc(val)}</span>"
        if key == "sentiment":
            c = _SENT_COLORS.get(val, "#9ca3af")
            return f"<span style='color:{c};font-weight:500'>{_esc(val)}</span>"
        if key == "priority":
            c = _PRIO_COLORS.get(val, "#9ca3af")
            return f"<span style='color:{c};font-weight:500'>{_esc(val)}</span>"
        if key == "message":
            return f"<span class='obs-message'>&ldquo;{_esc(val)}&rdquo;</span>"
        return f"<span style='color:#aaa'>{_esc(str(val))}</span>"

    rows = "".join(
        f"<tr><td class='obs-key'>{key}</td><td class='obs-val'>{_val(key, obs.get(key,'—'))}</td></tr>"
        for key in ["ticket_id", "issue_type", "sentiment", "priority", "message"]
    )
    history = obs.get("history", [])
    if history:
        hist_html = "".join(f"<div class='hist-mini'>{_esc(h)}</div>" for h in history[-5:])
        rows += f"<tr><td class='obs-key'>history</td><td class='obs-val'>{hist_html}</td></tr>"

    return f"""
<div class='pg-panel'>
  <div class='pg-panel-head'>Current Ticket</div>
  <div class='pg-panel-body'><table class='obs-table'><tbody>{rows}</tbody></table></div>
</div>"""


def _steps_html(steps: List[dict]) -> str:
    if not steps:
        return f"""
<div class='pg-panel'>
  <div class='pg-panel-head'>Phase Steps</div>
  <div class='pg-panel-body'><div class='empty-state'>No steps taken yet. Click Run Agent to start.</div></div>
</div>"""
    rows = ""
    for s in steps:
        pc  = _PHASE_COLORS.get(s["phase"], "#9ca3af")
        sc  = _score_color(s["score"])
        ac  = _ACTION_COLORS.get(s["action"], "#9ca3af")
        err = f"<span style='color:#ff6b6b;font-size:10px'> ⚠</span>" if s.get("error") else ""
        rows += f"""
<tr>
  <td class='tc'><code style='color:#555'>{s["step"]}</code></td>
  <td><span style='color:{pc};font-weight:600;font-size:11px'>{_esc(s["phase_name"])}</span></td>
  <td>{_badge(s["action"], ac, "11")}</td>
  <td><code style='color:#ffa94d;font-size:11px'>{_esc(s["team"])}</code></td>
  <td class='tr'><span style='color:{sc};font-weight:700'>{s["score"]:.3f}</span>{err}</td>
</tr>"""
    return f"""
<div class='pg-panel'>
  <div class='pg-panel-head'>Phase Steps</div>
  <div class='pg-panel-body'>
    <table class='data-table'>
      <thead><tr><th class='tc'>Step</th><th>Phase</th><th>Action</th><th>Team Routed</th><th class='tr'>Score</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</div>"""


def _score_html(rewards: List[float], avg: float, done: bool) -> str:
    if not rewards:
        return f"""
<div class='pg-panel'>
  <div class='pg-panel-head'>Episode Score</div>
  <div class='pg-panel-body'><div class='empty-state'>No scores yet.</div></div>
</div>"""
    ac  = _score_color(avg)
    dc  = "#00d084" if done else "#ffa94d"
    pills = "".join(
        f"<span class='score-pill' style='background:{_score_color(r)}1a;color:{_score_color(r)};border:1px solid {_score_color(r)}44'>{r:.3f}</span>"
        for r in rewards
    )
    return f"""
<div class='metrics-bar'>
  <div class='metric-cell'><div class='metric-label'>Avg Score</div>
    <div class='metric-val' style='color:{ac}'>{avg:.3f}</div></div>
  <div class='metric-cell'><div class='metric-label'>Steps</div>
    <div class='metric-val'>{len(rewards)}</div></div>
  <div class='metric-cell'><div class='metric-label'>Done</div>
    <div class='metric-val sm' style='color:{dc}'>{str(done).lower()}</div></div>
  <div class='metric-cell'><div class='metric-label'>Per-Phase</div>
    <div style='padding-top:6px'>{pills}</div></div>
</div>"""


def _team_html(team_lines: List[str]) -> str:
    if not team_lines:
        return f"""
<div class='pg-panel'>
  <div class='pg-panel-head'>Team Assignment</div>
  <div class='pg-panel-body'><div class='empty-state'>No team assigned yet.</div></div>
</div>"""
    ICONS_MAP = {
        "logistics_team":    "🚚", "tech_support_team": "🛠",
        "safety_team":       "⚠️", "finance_team":       "💳",
        "orders_team":       "📦", "management_team":    "👔",
    }
    cards = ""
    for line in team_lines:
        # line format: "[TKT-XXX]  Team Name — subtitle"
        parts   = line.split("  ", 1)
        tid_str = parts[0] if parts else line
        rest    = parts[1] if len(parts) > 1 else ""
        # find which team key matches
        matched_key = next((k for k in _TEAM_NAMES if _TEAM_NAMES[k] in rest), None)
        icon    = ICONS_MAP.get(matched_key, "📋")
        name    = _TEAM_NAMES.get(matched_key, rest.split(" — ")[0]) if matched_key else rest.split(" — ")[0]
        sub     = _TEAM_SUBTITLES.get(matched_key, rest.split(" — ", 1)[1] if " — " in rest else "")
        cards += f"""
<div class='team-card'>
  <div class='team-icon'>{icon}</div>
  <div>
    <div class='team-name'>{_esc(name)}</div>
    <div class='team-sub'><code style='font-size:11px;color:#444'>{_esc(tid_str)}</code> · {_esc(sub)}</div>
  </div>
</div>"""
    return f"""
<div class='pg-panel'>
  <div class='pg-panel-head'>Team Assignment</div>
  <div class='pg-panel-body' style='padding:12px'>{cards}</div>
</div>"""


def _msg_html(msg: str) -> str:
    if not msg:
        return f"""
<div class='pg-panel'>
  <div class='pg-panel-head'>Agent Response</div>
  <div class='pg-panel-body'><div class='empty-state'>No customer-facing message (triage or route phase).</div></div>
</div>"""
    return f"""
<div class='pg-panel'>
  <div class='pg-panel-head'>Agent Response</div>
  <div class='pg-panel-body' style='padding:16px'>
    <div class='agent-response'>{_esc(msg)}</div>
  </div>
</div>"""


def _history_html() -> str:
    if not env.history:
        return f"""
<div class='pg-panel'>
  <div class='pg-panel-head'>Action History</div>
  <div class='pg-panel-body'><div class='empty-state'>History will appear here after reset.</div></div>
</div>"""
    rows = ""
    for entry in env.history:
        is_sys = entry.startswith("System:") or entry.startswith("Customer:")
        cl = "#1f1f1f" if is_sys else "#60a5fa0d"
        bl = "#2a2a2a" if is_sys else "#60a5fa"
        fc = "#555"    if is_sys else "#ccc"
        rows += f"<div style='padding:8px 16px;border-bottom:1px solid #1a1a1a;border-left:3px solid {bl};background:{cl};font-size:12px;color:{fc};line-height:1.5'>{_esc(entry)}</div>"
    return f"""
<div class='pg-panel'>
  <div class='pg-panel-head'>Action History</div>
  <div class='pg-panel-body'>{rows}</div>
</div>"""

# ---------------------------------------------------------------------------
# Gradio event handlers
# ---------------------------------------------------------------------------

def ui_reset(task_choice: str):
    global _agent_history
    _agent_history = []
    task_name = {"Easy (Classification)": "easy", "Medium (Assignment)": "medium", "Hard (Full Resolution)": "hard"}.get(task_choice, "easy")
    res = env.reset(task_name)
    obs_json = res.observation.model_dump_json(indent=2)
    empty_score = "<div class='pg-panel'><div class='pg-panel-head'>Episode Score</div><div class='pg-panel-body'><div class='empty-state'>No scores yet.</div></div></div>"
    return (
        _obs_html(obs_json),
        _msg_html(""),
        _steps_html([]),
        empty_score,
        _history_html(),
        _team_html([]),
    )


async def ui_auto_step():
    global _agent_history

    if env.done:
        return (
            _obs_html(env.get_current_observation().model_dump_json(indent=2)),
            _msg_html(""),
            _steps_html([]),
            _score_html([], 0.0, True),
            _history_html(),
            _team_html([]),
        )

    all_steps:   List[dict]  = []
    all_rewards: List[float] = []
    last_message = ""
    team_lines:  List[str]   = []

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
                action = Action(action_type="respond", response="Thank you for contacting us. Our team will look into this shortly.")

        result  = env.step(action)
        reward  = result.reward
        done    = result.done

        _agent_history = env.history.copy()
        all_rewards.append(reward)

        all_steps.append({
            "step":       step_num,
            "phase":      current_phase,
            "phase_name": _PHASE_NAMES.get(current_phase, str(current_phase)),
            "action":     action.action_type,
            "team":       action.team or "—",
            "score":      reward,
            "error":      error_msg,
        })

        if action.team:
            name = _TEAM_NAMES.get(action.team, action.team)
            sub  = _TEAM_SUBTITLES.get(action.team, "")
            tid  = obs.ticket_id if hasattr(obs, "ticket_id") else "?"
            team_lines.append(f"[{tid}]  {name} — {sub}")

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
            _steps_html([]),
            _score_html([], 0.0, True),
            _history_html(),
            _team_html([]),
        )

    try:
        current_phase = env.phase
        action = Action(
            action_type=a_type,
            team=t_name or None,
            response=resp_text or None,
        )
        result = env.step(action)
        reward = result.reward
        done   = result.done
        _agent_history = env.history.copy()

        steps = [{
            "step":       env.step_count,
            "phase":      current_phase,
            "phase_name": _PHASE_NAMES.get(current_phase, str(current_phase)),
            "action":     a_type,
            "team":       t_name or "—",
            "score":      reward,
            "error":      None,
        }]

        team_lines = []
        if t_name:
            name = _TEAM_NAMES.get(t_name, t_name)
            sub  = _TEAM_SUBTITLES.get(t_name, "")
            if env.current_scenario:
                tid = env.current_scenario.ticket.ticket_id
            else:
                tid = "?"
            team_lines.append(f"[{tid}]  {name} — {sub}")

        return (
            _obs_html(result.observation.model_dump_json(indent=2)),
            _msg_html(resp_text or ""),
            _steps_html(steps),
            _score_html([reward], reward, done),
            _history_html(),
            _team_html(team_lines),
        )

    except Exception as exc:
        err_html = f"<div style='padding:16px;color:#ff6b6b;font-size:13px'>Error: {_esc(str(exc))}</div>"
        return (
            _obs_html(env.get_current_observation().model_dump_json(indent=2)),
            _msg_html(""),
            f"<div class='pg-panel'><div class='pg-panel-head'>Error</div><div class='pg-panel-body'>{err_html}</div></div>",
            "<div class='pg-panel'><div class='pg-panel-head'>Episode Score</div><div class='pg-panel-body'><div class='empty-state'>—</div></div></div>",
            _history_html(),
            _team_html([]),
        )

# ---------------------------------------------------------------------------
# Gradio layout
# ---------------------------------------------------------------------------

TASK_CHOICES = ["Easy (Classification)", "Medium (Assignment)", "Hard (Full Resolution)"]

with gr.Blocks(css=CSS, title="Customer Support Agent — OpenEnv") as demo:

    gr.HTML(_hero_html())

    with gr.Tabs():

        # ── Tab: Overview ─────────────────────────────────────────────
        with gr.Tab("Overview"):
            gr.HTML(_overview_html())

        # ── Tab: Scenarios ────────────────────────────────────────────
        with gr.Tab("Scenarios"):
            gr.HTML(_scenarios_html())

        # ── Tab: Playground ───────────────────────────────────────────
        with gr.Tab("Playground"):
            gr.HTML("<h2 class='section-h'>Live interactive episode</h2><p class='section-sub'>Select a tier, reset the environment, then run the full 3-phase episode automatically or step through manually.</p>")

            with gr.Row():
                task_dd   = gr.Dropdown(choices=TASK_CHOICES, value=TASK_CHOICES[0], label="Task Tier", scale=4)
                reset_btn = gr.Button("Reset Episode",         variant="secondary", scale=1, min_width=140)
                auto_btn  = gr.Button("Run Agent (Full Episode)", variant="primary",  scale=2, min_width=200)

            with gr.Row():
                with gr.Column(scale=1):
                    obs_out  = gr.HTML(_obs_html(""))
                    hist_out = gr.HTML(_history_html())

                with gr.Column(scale=1):
                    steps_out = gr.HTML(_steps_html([]))
                    team_out  = gr.HTML(_team_html([]))
                    msg_out   = gr.HTML(_msg_html(""))

            score_out = gr.HTML("<div class='pg-panel'><div class='pg-panel-head'>Episode Score</div><div class='pg-panel-body'><div class='empty-state'>No scores yet.</div></div></div>")

            with gr.Accordion("Manual Step Override", open=False):
                gr.HTML("<p style='font-size:12px;color:#555;padding:4px 0 12px'>Step through individual phases manually. Phase 1 requires classify, Phase 2 requires assign + team, Phase 3 requires respond/refund/escalate.</p>")
                with gr.Row():
                    act_type = gr.Dropdown(
                        choices=["classify", "assign", "respond", "refund", "escalate"],
                        value="classify", label="Action Type", scale=1,
                    )
                    act_team = gr.Textbox(label="Team (Phase 2 only)", scale=1, placeholder="e.g. tech_support_team")
                with gr.Row():
                    act_resp   = gr.Textbox(label="Response Text (Phase 3 only)", lines=2, scale=3,
                                            placeholder="Write the customer-facing response here...")
                    manual_btn = gr.Button("Submit Action", variant="primary", scale=1, min_width=140)

        # ── Tab: Try It ────────────────────────────────────────────────
        with gr.Tab("Try It"):
            gr.HTML(_try_it_html())

    gr.HTML("<div class='footer'>customer-support-agent · 3-phase MDP · 15 tickets · 6 specialist teams · deterministic grader · reward ∈ [0.002, 0.998]<br/><a href='https://github.com/AdityaK-labs/Support-Agent/tree/Tarun'>github</a> · <a href='https://huggingface.co/spaces/Tarun21W/MetaAI'>huggingface</a></div>")

    _outputs = [obs_out, msg_out, steps_out, score_out, hist_out, team_out]

    reset_btn.click( ui_reset,        inputs=[task_dd],                          outputs=_outputs)
    auto_btn.click(  ui_auto_step,    inputs=[],                                 outputs=_outputs)
    manual_btn.click(ui_manual_step,  inputs=[act_type, act_team, act_resp],     outputs=_outputs)

demo.queue()
app = gr.mount_gradio_app(app, demo, path="/")

def main():
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=True)

if __name__ == "__main__":
    main()