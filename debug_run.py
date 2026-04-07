import asyncio
import os
import json
from huggingface_hub import AsyncInferenceClient
from inference import build_user_prompt, call_api_model
from support_env import SupportEnvWrapper
from openenv.models import Action as SupportAction

async def debug():
    env = await SupportEnvWrapper.from_docker_image(None)
    env.env.task_name = 'easy'
    env.env.scenarios = [env.env.__class__('easy').scenarios[0]]
    
    res = await env.reset()
    obs = res.observation
    client = AsyncInferenceClient(model='Qwen/Qwen2.5-72B-Instruct', token=os.getenv('HF_TOKEN'))
    
    print(f"Start Obs: {obs}")
    history = []
    
    for i in range(5):
        print(f"\n--- Step {i+1} ---")
        prompt = build_user_prompt(i+1, obs, history)
        try:
            data = await call_api_model({"message": obs.message, "sentiment": obs.sentiment}, client, prompt)
            print(f"API Data: {json.dumps(data, indent=2)}")
            
            act = SupportAction(**data)
            print(f"Parsed Action: {act}")
            
            res = await env.step(act)
            obs = res.observation
            feedback_str = getattr(res, "info", {}).get("feedback", "") if isinstance(getattr(res, "info", None), dict) else ""
            log_line = f"Reward: {res.reward}, Done: {res.done}, Feedback: {feedback_str}\n"
            
            with open("debug_log.txt", "a") as f:
                f.write(f"\n--- Step {i+1} ---\n")
                f.write(f"API Data: {json.dumps(data, indent=2)}\n")
                f.write(f"Parsed Action: {act}\n")
                f.write(log_line)
            
            print(log_line)
            
            if res.done:
                break
        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}")
            break

if __name__ == "__main__":
    asyncio.run(debug())
