import asyncio
import os
import subprocess
import sys
from dotenv import load_dotenv

load_dotenv()

def run_inference_for_task(task_name):
    print(f"\n{'='*50}\nRunning Baseline for Task: {task_name.upper()}\n{'='*50}")
    env_vars = os.environ.copy()
    env_vars["SUPPORT_ENV_TASK"] = task_name
    
    script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "inference.py")
    
    result = subprocess.run([sys.executable, script_path], env=env_vars, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"Error ({task_name}):", result.stderr)

if __name__ == "__main__":
    tasks = ["easy", "medium", "hard"]
    for task in tasks:
        run_inference_for_task(task)
