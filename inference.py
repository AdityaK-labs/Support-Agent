import asyncio
import json
import os
import re
import textwrap
from typing import List, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

from support_env import SupportEnvWrapper, SupportAction

IMAGE_NAME = os.getenv("IMAGE_NAME")
API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")   # HF_TOKEN is primary per spec
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("SUPPORT_ENV_TASK", "easy")
BENCHMARK = os.getenv("SUPPORT_ENV_BENCHMARK", "support_env")

MAX_STEPS = 5
TEMPERATURE = 0.2
MAX_TOKENS = 500
SUCCESS_SCORE_THRESHOLD = 0.5 

_MAX_REWARD_PER_STEP = 1.0
MAX_TOTAL_REWARD = MAX_STEPS * _MAX_REWARD_PER_STEP

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an autonomous customer support agent for a large e-commerce and SaaS company.
    Read the customer ticket carefully and take exactly ONE action to resolve it.

    STANDARD OPERATING PROCEDURE (SOP):

    ACTION DECISION RULES - follow strictly in order:

    1. USE classify
       - When the issue type is 'unknown' and just needs categorization (simple address changes, account updates).
       - Do NOT include team or response.

    2. USE assign + correct TEAM
       - When the ticket requires specialist handling.
       - Always set 'team' to EXACTLY one from this directory:
         * logistics_team      -> Lost packages, shipping tracking, delivery disputes
         * tech_support_team   -> Login issues, password resets, software bugs, API errors, data loss
         * safety_team         -> Product defects, overheating, recalls, safety hazards
         * finance_team        -> Invoice errors, billing corrections, tax/payment issues
         * orders_team         -> Bulk orders, corporate accounts, order modifications
         * management_team     -> Escalated complaints, refund delays, manager requests

    3. USE escalate + correct TEAM
       - When the customer is extremely angry, threatening, or explicitly requests a manager.
       - When a severe safety issue is involved.
       - Set team to management_team (or safety_team for hazards).
       - Always include both 'team' and 'response'.
       - PUNISHMENT: Using escalate when NOT required costs -0.30 score.

    4. USE refund
       - ONLY when a product was definitively broken on arrival or provably the company's fault.
       - Always include a 'response' explaining the refund.
       - PUNISHMENT: Issuing refund when NOT required costs -0.50 score.

    5. USE respond
       - When the customer needs information: policy, pricing, features, or technical details.
       - Include a detailed 'response' addressing their specific question with exact details from their message.

    PUNISHMENT SUMMARY:
    - Unnecessary refund:     -0.50
    - Unnecessary escalation: -0.30
    - Wrong team assigned:    -0.15
    - More than 3 steps:      -0.10 per extra step

    OUTPUT FORMAT - MANDATORY:
    Respond with ONLY a raw JSON object. No markdown, no explanation, no code fences.
    Schema:
    {"action_type": "...", "team": "...", "response": "..."}

    Examples:
    {"action_type": "classify"}
    {"action_type": "assign", "team": "tech_support_team"}
    {"action_type": "escalate", "team": "management_team", "response": "I sincerely apologize for the experience. I am escalating order #12345 to our management team immediately."}
    {"action_type": "respond", "response": "Our API rate limits reset hourly. The 429 error means you have exceeded the limit for that hour. Please wait 60 minutes or contact us to discuss a higher plan."}
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def build_user_prompt(step: int, obs: any, history: List[str]) -> str:
    obs_dict = {
        "ticket_id": obs.ticket_id,
        "issue_type": obs.issue_type,
        "sentiment": obs.sentiment,
        "priority": obs.priority,
        "message": obs.message,
    }
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Ticket Context: {json.dumps(obs_dict, indent=2)}
        Previous steps history:
        {history_block}
        
        Send your JSON action.
        """
    ).strip()

def fallback_policy(state: dict) -> dict:
    msg = state["message"].lower()

    # Refund cases
    if "refund" in msg or "money" in msg:
        return {
            "action_type": "refund",
            "team": "billing",
            "response": "We have processed your refund. Apologies for the inconvenience."
        }

    # Technical issues
    elif "login" in msg or "error" in msg or "bug" in msg:
        return {
            "action_type": "assign",
            "team": "tech",
            "response": "Our technical team is looking into your issue."
        }

    # Angry customer
    elif state["sentiment"] == "angry":
        return {
            "action_type": "escalate",
            "team": "priority_support",
            "response": "We are escalating your issue for immediate attention."
        }

    # Default
    return {
        "action_type": "classify",
        "team": "general",
        "response": "Thank you for contacting support."
    }

async def call_api_model(state: dict, client: AsyncOpenAI, user_prompt: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    
    text = response.choices[0].message.content.strip()
    
    # Robust Regex Extraction to ignore markdown/chatty text padding
    json_match = re.search(r"(\{.*\})", text, re.DOTALL)
    if json_match:
        clean_json = json_match.group(1)
        try:
            return json.loads(clean_json)
        except json.JSONDecodeError as decode_exc:
            raise ValueError(f"JSON decode failed. Response: {text}") from decode_exc
    else:
        raise ValueError(f"No JSON object detected in response. Raw response: {text}")

async def get_action(state: dict, client: AsyncOpenAI, user_prompt: str) -> SupportAction:
    # API errors (network, auth, rate-limit) must NOT be silently swallowed — they
    # indicate a misconfigured proxy and must surface so the episode fails visibly.
    # Only fall back on JSON parse/decode issues where the API was reached but returned
    # malformed output.
    try:
        data = await call_api_model(state, client, user_prompt)
    except (ValueError, json.JSONDecodeError) as parse_exc:
        # API was called but response was unparseable — use fallback heuristic
        print(f"\n[DEBUG] ⚠️ JSON parse failed: {type(parse_exc).__name__}: {parse_exc}\n", flush=True)
        fb_dict = fallback_policy(state)
        return SupportAction(**fb_dict)
    # All other exceptions (openai.APIConnectionError, openai.AuthenticationError,
    # httpx errors, etc.) propagate — they will be caught in main() and abort the run
    # with a visible error, preventing silent fallback-only episodes.
    return SupportAction(**data)

async def run_episode(task_name: str, client: AsyncOpenAI) -> None:
    """Run one full episode for the given task and emit START/STEP/END logs."""
    env = await SupportEnvWrapper.from_docker_image(IMAGE_NAME)
    env.env.task_name = task_name

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            user_prompt = build_user_prompt(step, obs, history)
            state_dict = {
                "message": obs.message,
                "sentiment": obs.sentiment,
            }
            action = await get_action(state_dict, client, user_prompt)
            action_str = json.dumps(action.model_dump())

            result = await env.step(action)
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=None)
            history.append(f"Step {step}: {action_str} -> reward {reward:+.2f}")

            if done:
                break

        score = sum(rewards) / float(len(rewards)) if rewards else 0.0
        score = min(max(score, 0.002), 0.998)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close() error: {exc}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    if not API_BASE_URL:
        raise RuntimeError("[FATAL] API_BASE_URL is not set. The LLM proxy URL must be provided via environment variable.")
    if not API_KEY:
        raise RuntimeError("[FATAL] HF_TOKEN (or API_KEY) is not set. A valid API key must be provided via environment variable.")

    print(f"[INFO] Using API_BASE_URL={API_BASE_URL} MODEL_NAME={MODEL_NAME}", flush=True)
    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # If a specific task is requested run only that one, otherwise run all three
    # so the validator can enumerate tasks and verify each grader.
    if TASK_NAME in ("easy", "medium", "hard"):
        tasks_to_run = [TASK_NAME]
    else:
        tasks_to_run = ["easy", "medium", "hard"]

    for task in tasks_to_run:
        await run_episode(task, client)


if __name__ == "__main__":
    asyncio.run(main())

