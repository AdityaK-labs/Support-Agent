import random
from typing import Dict, Any, Optional

from openenv.models import Observation, Action, Reward, EnvState, StepResult, ResetResult
from openenv.tasks import TASK_REGISTRY
from openenv.reward import compute_reward

# Phase labels for logging and prompting
PHASE_LABELS = {
    1: "triage",
    2: "route",
    3: "resolve",
}

PHASE_INSTRUCTIONS = {
    1: (
        "PHASE 1 — TRIAGE: Read the ticket and classify it.\n"
        "Required action: {\"action_type\": \"classify\"}"
    ),
    2: (
        "PHASE 2 — ROUTE: Assign the ticket to the correct specialist team.\n"
        "Required action: {\"action_type\": \"assign\", \"team\": \"<team_name>\"}\n"
        "Teams: logistics_team | tech_support_team | safety_team | finance_team | orders_team | management_team"
    ),
    3: (
        "PHASE 3 — RESOLVE: Take the final resolution action.\n"
        "Choose ONE of:\n"
        "  escalate — angry/manager request/safety emergency (include team + response)\n"
        "  refund   — provably company fault/wrong product (include response)\n"
        "  respond  — customer needs information (write detailed response)"
    ),
}


class SupportEnv:
    def __init__(self, task_name: str = "easy"):
        self.task_name = task_name
        self.scenarios = TASK_REGISTRY.get(task_name, TASK_REGISTRY["easy"])
        self.current_scenario = None
        self.step_count = 0
        self.phase = 1
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
        self.phase = 1
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
        gt = self.current_scenario.ground_truth

        # Log action to history
        act_str = f"[Phase {self.phase}/{PHASE_LABELS[self.phase]}] Agent: {action.action_type}"
        if action.team:
            act_str += f" → {action.team}"
        if action.response:
            act_str += f" | \"{action.response[:80]}{'...' if len(action.response) > 80 else ''}\""
        self.history.append(act_str)

        # Grade this step against its phase
        reward = compute_reward(action, gt, self.step_count, phase=self.phase)
        self.total_reward += reward.score

        # Advance phase and update observation state
        if self.phase == 1:
            # After triage: reveal the true issue_type so phase 2 routing is informed
            self.current_scenario.ticket.issue_type = gt.issue_type
            self.history.append(
                f"System: Issue classified as '{gt.issue_type}'. Proceed to route the ticket."
            )
            self.phase = 2

        elif self.phase == 2:
            # After routing: record which team was assigned; prepare for resolution
            assigned = action.team or "(no team)"
            self.history.append(
                f"System: Ticket routed to '{assigned}'. Proceed to resolve the customer's issue."
            )
            self.phase = 3

        elif self.phase == 3:
            # Resolution phase — episode terminates here
            if action.action_type == "respond" and reward.score < 0.5:
                # Poor response: customer is still unhappy, allow one retry (up to max_steps)
                self.history.append(
                    "Customer: I'm still not satisfied. Could you be more specific?"
                )
                # Stay in phase 3 for retry
            else:
                self.done = True
                if action.action_type in ("refund", "escalate"):
                    self.history.append("System: Ticket escalated/refunded and closed.")
                else:
                    self.history.append("Customer: Thank you, that answers my question!")

        # Hard cap at 5 steps
        if self.step_count >= 5:
            self.done = True

        obs = self.get_current_observation()

        return StepResult(
            observation=obs,
            reward=reward.score,
            done=self.done,
            info={"feedback": reward.feedback, "phase": self.phase}
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
            return Observation(
                ticket_id="None",
                issue_type="None",
                sentiment="None",
                priority="None",
                message="No active scenario. Please reset.",
                history=self.history.copy()
            )
        obs = self.current_scenario.ticket.copy(deep=True)
        obs.history = self.history.copy()
        return obs

    def get_phase_instruction(self) -> str:
        return PHASE_INSTRUCTIONS.get(self.phase, PHASE_INSTRUCTIONS[3])

    def close(self):
        pass

    @classmethod
    def from_docker_image(cls, image_name: str = None):
        return cls()