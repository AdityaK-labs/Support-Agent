import random
from typing import Dict, Any, Optional

from openenv.models import Observation, Action, Reward, EnvState, StepResult, ResetResult
from openenv.tasks import TASK_REGISTRY
from openenv.reward import compute_reward

class SupportEnv:
    def __init__(self, task_name: str = "easy"):
        self.task_name = task_name
        self.scenarios = TASK_REGISTRY.get(task_name, TASK_REGISTRY["easy"])
        self.current_scenario = None
        self.step_count = 0
        self.history = []
        self.done = False
        self.total_reward = 0.0
        self.reset(task_name)
        
    def reset(self, task_name: Optional[str] = None) -> ResetResult:
        if task_name:
            self.task_name = task_name
            self.scenarios = TASK_REGISTRY.get(task_name, TASK_REGISTRY["easy"])
            
        self.current_scenario = random.choice(self.scenarios)
        self.step_count = 0
        self.history = []
        self.done = False
        self.total_reward = 0.0
        
        obs = self.current_scenario.ticket.copy(deep=True)
        obs.history = self.history.copy()
        
        return ResetResult(observation=obs, done=False, info={})
        
    def step(self, action: Action) -> StepResult:
        if self.done:
            return StepResult(
                observation=self.get_current_observation(),
                reward=0.0,
                done=True,
                info={"error": "Episode already done."}
            )
            
        self.step_count += 1
        
        # log action to history
        act_str = f"Agent Action: {action.action_type}"
        if action.team: act_str += f" -> Team: {action.team}"
        if action.response: act_str += f" | Response: {action.response}"
        self.history.append(act_str)
        
        reward = compute_reward(action, self.current_scenario.ground_truth, self.step_count)
        self.total_reward += reward.score
        
        # Determine if done
        # Hard terminate after 5 steps
        if self.step_count >= 5:
            self.done = True
        elif action.action_type in ["classify", "assign", "refund", "escalate"]:
            # These are terminal internal routing / finance actions
            self.done = True
            self.history.append("System: Ticket successfully processed and closed.")
        elif action.action_type == "respond":
            # Real-world simulation: does the response satisfy the user?
            if reward.score >= 0.8:
                self.done = True
                self.history.append("Customer: Thank you, that resolves my issue!")
            else:
                # Keep ticket open, simulated unhappy customer
                self.history.append("Customer: I am still confused. This doesn't actually solve my problem. Could you explain better?")
            
        obs = self.get_current_observation()
        
        return StepResult(
            observation=obs,
            reward=reward.score,
            done=self.done,
            info={"feedback": reward.feedback}
        )
        
    def state(self) -> EnvState:
        return EnvState(
            observation=self.get_current_observation(),
            step_count=self.step_count,
            done=self.done,
            episode_id="local_run",
            total_reward=self.total_reward,
            task_name=self.task_name,
            max_steps=5
        )
        
    def get_current_observation(self) -> Observation:
        if not self.current_scenario:
            return Observation(ticket_id="None", issue_type="None", sentiment="None", priority="None", message="No active scenario. Please reset.", history=self.history.copy())
        obs = self.current_scenario.ticket.copy(deep=True)
        obs.history = self.history.copy()
        return obs
        
    def close(self):
        pass
        
    @classmethod
    def from_docker_image(cls, image_name: str = None):
        # Dummy factory for the inference script compat
        return cls()
