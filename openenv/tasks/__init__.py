"""Task definitions for easy, medium, and hard difficulty levels."""
from openenv.tasks.easy import EASY_TASKS
from openenv.tasks.medium import MEDIUM_TASKS
from openenv.tasks.hard import HARD_TASKS

TASK_REGISTRY = {
    "easy": EASY_TASKS,
    "medium": MEDIUM_TASKS,
    "hard": HARD_TASKS,
}

__all__ = ["EASY_TASKS", "MEDIUM_TASKS", "HARD_TASKS", "TASK_REGISTRY"]
