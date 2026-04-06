import asyncio
import os
import json
import textwrap
from typing import List, Optional

from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()

from support_env import SupportEnvWrapper, SupportAction

IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
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
    You are an autonomous customer support agent.
    You will receive details about a customer ticket.
    Your goal is to choose the correct action to resolve the ticket.
    Available actions: classify, assign, respond, refund, escalate.
    If assigning or escalating, you must provide a 'team'.
    If responding or refunding, you must provide a 'response'.
    
    You MUST output valid JSON only, matching this schema exactly:
    {
      "action_type": "classify" | "assign" | "respond" | "refund" | "escalate",
      "team": "string (optional)",
      "response": "string (optional)"
    }
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

def call_local_model(state: dict, model, tokenizer, user_prompt: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=MAX_TOKENS, temperature=TEMPERATURE, do_sample=True)
    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())

def get_action(state: dict, model, tokenizer, user_prompt: str) -> SupportAction:
    try:
        data = call_local_model(state, model, tokenizer, user_prompt)
        return SupportAction(**data)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        fb_dict = fallback_policy(state)
        return SupportAction(**fb_dict)

async def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    env = await SupportEnvWrapper.from_docker_image(IMAGE_NAME)
    env.env.task_name = TASK_NAME

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()
        obs = result.observation
        
        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            user_prompt = build_user_prompt(step, obs, history)
            state_dict = {
                "message": obs.message,
                "sentiment": obs.sentiment
            }
            action = get_action(state_dict, model, tokenizer, user_prompt)
            action_str = json.dumps(action.model_dump())

            # environment step
            result = await env.step(action)
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)
            history.append(f"Step {step}: {action_str} -> reward {reward:+.2f}")

            if done:
                break

        score = sum(rewards) / float(len(rewards)) if rewards else 0.0 
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())
