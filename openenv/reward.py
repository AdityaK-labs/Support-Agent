from openenv.models import Action, GroundTruth, Reward
from openenv.graders.grader import GraderEngine

def compute_reward(action: Action, ground_truth: GroundTruth, step_count: int) -> Reward:
    """Computes the reward for a given action against the ground truth."""
    return GraderEngine.grade(action, ground_truth, step_count)
