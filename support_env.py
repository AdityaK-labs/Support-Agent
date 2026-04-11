from openenv.env import SupportEnv
from openenv.models import Action as SupportAction

# Mapping OpenEnv structure to the expected wrapper format from the prompt.
class SupportEnvWrapper:
    def __init__(self, task="easy"):
        self.env = SupportEnv(task_name=task)
        
    async def reset(self):
        # We need to return an object with `.observation` and `.done` per inference.py
        result = self.env.reset()
        class WrappedResult:
            def __init__(self, obs, done):
                self.observation = obs
                self.done = done
        return WrappedResult(obs=result.observation, done=result.done)
        
    async def step(self, action):
        result = self.env.step(action)
        class WrappedResult:
            def __init__(self, obs, reward, done):
                self.observation = obs
                self.reward = reward
                self.done = done
        return WrappedResult(obs=result.observation, reward=result.reward, done=result.done)

    async def close(self):
        self.env.close()

    @classmethod
    async def from_docker_image(cls, image_name: str = None, task: str = "easy"):
        return cls(task=task)
