"""
Multi-episode benchmark runner for the Support Agent.
Runs multiple episodes across easy/medium/hard tasks and prints a summary table.
"""
import asyncio
import os
import json
import textwrap
from typing import List, Optional

from dotenv import load_dotenv
from huggingface_hub import AsyncInferenceClient

load_dotenv()

from support_env import SupportEnvWrapper, SupportAction
from openenv.tasks import TASK_REGISTRY
from inference import (
    SYSTEM_PROMPT, MAX_STEPS, MAX_TOKENS, TEMPERATURE,
    SUCCESS_SCORE_THRESHOLD, build_user_prompt, fallback_policy,
    get_action, API_KEY
)

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
EPISODES_PER_TASK = 5
TASKS = ["easy", "medium", "hard"]


async def run_episode(client: AsyncInferenceClient, task_name: str, episode_num: int):
    """Run a single episode with a fresh env and return results."""
    env = await SupportEnvWrapper.from_docker_image(None)
    env.env.task_name = task_name
    env.env.scenarios = TASK_REGISTRY.get(task_name, TASK_REGISTRY["easy"])

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0

    result = await env.reset()
    obs = result.observation

    for step in range(1, MAX_STEPS + 1):
        if result.done:
            break

        user_prompt = build_user_prompt(step, obs, history)
        state_dict = {"message": obs.message, "sentiment": obs.sentiment}
        action = await get_action(state_dict, client, user_prompt)
        action_str = json.dumps(action.model_dump())

        result = await env.step(action)
        obs = result.observation
        reward = result.reward or 0.0
        done = result.done

        rewards.append(reward)
        steps_taken = step
        history.append(f"Step {step}: {action_str} -> reward {reward:+.2f}")

        if done:
            break

    await env.close()

    score = sum(rewards) / float(len(rewards)) if rewards else 0.0
    score = min(max(score, 0.0), 1.0)
    success = score >= SUCCESS_SCORE_THRESHOLD

    return {
        "task": task_name,
        "episode": episode_num,
        "steps": steps_taken,
        "score": score,
        "success": success,
        "rewards": rewards,
    }


async def main():
    print(f"\n{'='*60}")
    print(f"  BENCHMARK: {MODEL_NAME}")
    print(f"  Episodes per task: {EPISODES_PER_TASK}")
    print(f"  Tasks: {', '.join(TASKS)}")
    print(f"{'='*60}\n")

    if not API_KEY:
        print("[WARNING] No HF_TOKEN found! Remote inference will likely fail.")
        
    print("[*] Initializing remote client...")
    client = AsyncInferenceClient(model=MODEL_NAME, token=API_KEY)
    print(f"[*] Client connected to {MODEL_NAME}!\n")

    all_results = []

    for task in TASKS:
        print(f"\n--- Task: {task.upper()} ---")
        for ep in range(1, EPISODES_PER_TASK + 1):
            result = await run_episode(client, task, ep)
            all_results.append(result)
            status = "PASS" if result["success"] else "FAIL"
            rewards_str = ",".join(f"{r:.2f}" for r in result["rewards"])
            print(
                f"  Episode {ep}: score={result['score']:.3f}  "
                f"steps={result['steps']}  {status}  "
                f"rewards=[{rewards_str}]"
            )

    # Summary table
    print(f"\n\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"{'Task':<10} {'Avg Score':<12} {'Success Rate':<15} {'Avg Steps':<12}")
    print(f"{'-'*49}")

    overall_scores = []
    overall_successes = []

    for task in TASKS:
        task_results = [r for r in all_results if r["task"] == task]
        avg_score = sum(r["score"] for r in task_results) / len(task_results)
        success_rate = sum(1 for r in task_results if r["success"]) / len(task_results)
        avg_steps = sum(r["steps"] for r in task_results) / len(task_results)
        overall_scores.append(avg_score)
        overall_successes.extend([r["success"] for r in task_results])
        print(
            f"{task:<10} {avg_score:<12.3f} {success_rate*100:<15.1f}% {avg_steps:<12.1f}"
        )

    total_avg = sum(overall_scores) / len(overall_scores)
    total_success = sum(1 for s in overall_successes if s) / len(overall_successes)
    print(f"{'-'*49}")
    print(f"{'OVERALL':<10} {total_avg:<12.3f} {total_success*100:<15.1f}%")
    print(f"{'='*60}\n")
    
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    try:
        await env.close()
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(main())
