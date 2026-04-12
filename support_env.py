from dataclasses import dataclass
from typing import Any

from openenv.env import SupportEnv
from openenv.models import Action as SupportAction


@dataclass
class ResetResult:
    observation: Any
    done: bool


@dataclass
class StepResult:
    observation: Any
    reward: float
    done: bool
    phase: int  # phase AFTER the transition (what the env is now waiting for)


class SupportEnvWrapper:
    def __init__(self, task: str = "easy"):
        self.env = SupportEnv(task_name=task)

    async def reset(self) -> ResetResult:
        result = self.env.reset()
        return ResetResult(observation=result.observation, done=result.done)

    async def step(self, action: SupportAction) -> StepResult:
        result = self.env.step(action)
        return StepResult(
            observation=result.observation,
            reward=result.reward,
            done=result.done,
            phase=self.env.phase,  # actual phase from env, not a counter
        )

    async def close(self) -> None:
        self.env.close()

    @classmethod
    async def from_docker_image(cls, image_name: str = None, task: str = "easy") -> "SupportEnvWrapper":
        return cls(task=task)
