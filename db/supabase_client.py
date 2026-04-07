import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

class DummyClient:
    def table(self, *args, **kwargs): return self
    def insert(self, *args, **kwargs): return self
    def execute(self, *args, **kwargs): return None
    def select(self, *args, **kwargs): return self

def get_supabase() -> Client:
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            return create_client(SUPABASE_URL, SUPABASE_KEY)
        except Exception as e:
            print(f"Failed to connect to Supabase: {e}")
            return DummyClient()
    return DummyClient()

supabase = get_supabase()

def log_episode(task_name: str, total_reward: float, steps: int):
    try:
        supabase.table("episodes").insert({
            "task_name": task_name,
            "total_reward": total_reward,
            "steps": steps
        }).execute()
    except Exception as e:
        print("Warning: Could not log to Supabase", e)
