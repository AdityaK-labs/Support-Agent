from openenv.models import Action, GroundTruth, Reward
from openenv.graders.grader import GraderEngine

def compute_reward(action: Action, ground_truth: GroundTruth, step_count: int, phase: int = 3) -> Reward:
    """Computes the reward for a given action against the ground truth."""
    return GraderEngine.grade_phase(action, ground_truth, phase=phase, step_count=step_count)