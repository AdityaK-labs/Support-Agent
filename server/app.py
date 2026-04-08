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

# ---------------------------------------------------------------------------
# Required OpenEnv runtime endpoints
# ---------------------------------------------------------------------------

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
# Agent LLM logic (shared with inference.py)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are an autonomous customer support agent for a large e-commerce and SaaS company.
    Read the customer ticket carefully and take exactly ONE action to resolve it.

    ACTION DECISION RULES:
    1. classify  — ticket just needs categorisation (unknown issue type, no team needed)
    2. assign    — needs specialist: logistics_team, tech_support_team, safety_team,
                   finance_team, orders_team, management_team
    3. escalate  — extremely angry customer, safety hazard, or explicit manager request.
                   Always include team + response. PENALTY -0.30 if unnecessary.
    4. refund    — product broken on arrival / provably company fault.
                   Always include response. PENALTY -0.50 if unnecessary.
    5. respond   — customer needs information; write a detailed, specific response.

    OUTPUT FORMAT — MANDATORY:
    Raw JSON only. No markdown, no explanation.
    {"action_type": "...", "team": "...", "response": "..."}
""").strip()


def _build_prompt(obs, history: List[str]) -> str:
    ctx = {
        "ticket_id":  obs.ticket_id,
        "issue_type": obs.issue_type,
        "sentiment":  obs.sentiment,
        "priority":   obs.priority,
        "message":    obs.message,
    }
    hist = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(f"""
        Ticket: {json.dumps(ctx, indent=2)}
        History:
        {hist}
        Decide your action and return JSON.
    """).strip()


async def _call_llm(obs, history: List[str]) -> dict:
    if not API_BASE_URL or not API_KEY:
        raise RuntimeError("API_BASE_URL / HF_TOKEN not configured.")
    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    prompt = _build_prompt(obs, history)
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
# Gradio UI helpers
# ---------------------------------------------------------------------------

TASK_MAP = {
    "Easy (Classification)":  "easy",
    "Medium (Assignment)":    "medium",
    "Hard (Full Resolution)": "hard",
}

_agent_history: List[str] = []   # per-episode history for the agent


def ui_reset(task_choice: str):
    global _agent_history
    _agent_history = []
    task_name = TASK_MAP.get(task_choice, "easy")
    res = env.reset(task_name)
    obs_json = res.observation.model_dump_json(indent=2)
    return (
        obs_json,           # observation box
        "",                 # agent message box
        "",                 # action info box
        "",                 # reward box
        "",                 # history box
    )


async def ui_auto_step():
    """Run the full episode: loop until done, showing all steps taken."""
    global _agent_history

    if env.done:
        return (
            env.get_current_observation().model_dump_json(indent=2),
            "",
            "Episode finished — please reset.",
            "",
            _fmt_history(),
        )

    all_steps = []       # list of per-step summaries
    all_rewards = []
    last_message = ""

    MAX_UI_STEPS = 5

    for step_num in range(1, MAX_UI_STEPS + 1):
        if env.done:
            break

        obs = env.get_current_observation()

        try:
            action_data = await _call_llm(obs, _agent_history)
            action = Action(**action_data)
            error_msg = None
        except Exception as exc:
            error_msg = str(exc)
            action = Action(action_type="classify")

        result = env.step(action)
        reward = result.reward
        done   = result.done

        _agent_history = env.history.copy()
        all_rewards.append(reward)

        step_line = f"Step {step_num}: {action.action_type}"
        if action.team:
            step_line += f" → {action.team}"
        step_line += f"  |  score: {reward:.3f}"
        if error_msg:
            step_line += f"  [error: {error_msg}]"
        all_steps.append(step_line)

        if action.response:
            last_message = action.response

        if done:
            break

    act_summary = "\n".join(all_steps)
    avg_score = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    reward_text = (
        f"Steps taken: {len(all_rewards)}\n"
        f"Rewards:     {', '.join(f'{r:.3f}' for r in all_rewards)}\n"
        f"Avg score:   {avg_score:.3f}\n"
        f"Done:        {env.done}"
    )
    agent_message = last_message or "(no customer-facing message — classify/assign task)"

    return (
        env.get_current_observation().model_dump_json(indent=2),
        agent_message,
        act_summary,
        reward_text,
        _fmt_history(),
    )


def ui_manual_step(a_type: str, t_name: str, resp_text: str):
    """Manual step — user picks action, team, response."""
    global _agent_history

    if env.done:
        return (
            env.get_current_observation().model_dump_json(indent=2),
            "",
            "Episode finished — please reset.",
            "",
            _fmt_history(),
        )

    try:
        action = Action(
            action_type=a_type,
            team=t_name or None,
            response=resp_text or None,
        )
        result  = env.step(action)
        reward  = result.reward
        done    = result.done

        act_summary = f"Action:  {a_type}"
        if t_name:
            act_summary += f"\nTeam:    {t_name}"

        agent_message = resp_text or "(no response text provided)"

        reward_text = (
            f"Score:    {reward:.3f}\n"
            f"Done:     {done}\n"
            f"Feedback: {result.info.get('feedback', '')}"
        )

        _agent_history = env.history.copy()

        return (
            result.observation.model_dump_json(indent=2),
            agent_message,
            act_summary,
            reward_text,
            _fmt_history(),
        )

    except Exception as exc:
        return (
            env.get_current_observation().model_dump_json(indent=2),
            "",
            f"Error: {exc}",
            "",
            _fmt_history(),
        )


def _fmt_history() -> str:
    return "\n".join(f"  {h}" for h in env.history) if env.history else "(empty)"

# ---------------------------------------------------------------------------
# Gradio layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="OpenEnv Support Agent", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# OpenEnv — Autonomous Customer Support Agent")
    gr.Markdown(
        "Select a task level and hit **Reset**. "
        "Use **Run Agent Step** to let the LLM decide automatically, "
        "or expand **Manual Step** to override."
    )

    # ── Top bar ──────────────────────────────────────────────────────────────
    with gr.Row():
        task_dd  = gr.Dropdown(
            choices=list(TASK_MAP.keys()),
            value="Easy (Classification)",
            label="Task Level",
            scale=2,
        )
        reset_btn = gr.Button("Reset Environment", variant="secondary", scale=1)

    # ── Main columns ─────────────────────────────────────────────────────────
    with gr.Row():

        # Left — observation + history
        with gr.Column(scale=1):
            gr.Markdown("### Current Observation")
            obs_box = gr.Code(language="json", label="Ticket JSON", lines=14)

            gr.Markdown("### Action History")
            hist_box = gr.Textbox(label="", lines=6, interactive=False)

        # Right — agent output
        with gr.Column(scale=1):
            gr.Markdown("### Agent")

            auto_btn = gr.Button("Run Agent Step", variant="primary", size="lg")

            msg_box = gr.Textbox(
                label="Agent Message / Response",
                lines=5,
                interactive=False,
                placeholder="The agent's reply to the customer will appear here...",
            )
            act_box = gr.Textbox(
                label="Action Taken",
                lines=3,
                interactive=False,
            )
            reward_box = gr.Textbox(
                label="Score & Feedback",
                lines=4,
                interactive=False,
            )

            # ── Manual override (collapsed) ──────────────────────────────────
            with gr.Accordion("Manual Step (override)", open=False):
                act_type = gr.Dropdown(
                    choices=["classify", "assign", "respond", "refund", "escalate"],
                    value="classify",
                    label="Action Type",
                )
                act_team = gr.Textbox(label="Team (optional)")
                act_resp = gr.Textbox(label="Response text (optional)", lines=3)
                manual_btn = gr.Button("Step with Manual Action")

    # ── Wire up events ────────────────────────────────────────────────────────
    _outputs = [obs_box, msg_box, act_box, reward_box, hist_box]

    reset_btn.click(ui_reset,       inputs=[task_dd],                              outputs=_outputs)
    auto_btn.click( ui_auto_step,   inputs=[],                                     outputs=_outputs)
    manual_btn.click(ui_manual_step, inputs=[act_type, act_team, act_resp],        outputs=_outputs)

demo.queue()
app = gr.mount_gradio_app(app, demo, path="/")

def main():
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=True)

if __name__ == "__main__":
    main()