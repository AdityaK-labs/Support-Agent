"""
Typed Pydantic models for the OpenEnv Customer Support Environment.

Defines Observation, Action, Reward, EnvState, StepResult, TicketScenario,
and GroundTruth — the complete data contract for the environment.
"""

from __future__ import annotations

import enum
from typing import Any, Dict, List, Optional
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Enums for constrained fields
# ---------------------------------------------------------------------------

class ActionType(str, enum.Enum):
    CLASSIFY = "classify"
    ASSIGN = "assign"
    RESPOND = "respond"
    REFUND = "refund"
    ESCALATE = "escalate"


class Sentiment(str, enum.Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    ANGRY = "angry"


class Priority(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Core OpenEnv Models
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """What the agent sees at each step."""
    ticket_id: str = Field(..., description="Unique ticket identifier")
    issue_type: str = Field(..., description="Category of the customer issue")
    sentiment: str = Field(..., description="Customer sentiment: positive|neutral|negative|angry")
    priority: str = Field(..., description="Ticket priority: low|medium|high|critical")
    message: str = Field(..., description="The customer's message text")
    history: List[str] = Field(default_factory=list, description="Previous interaction history")


class Action(BaseModel):
    """What the agent does at each step."""
    action_type: str = Field(
        ...,
        description="Action to take: classify|assign|respond|refund|escalate",
    )
    team: Optional[str] = Field(
        None,
        description="Target team for assignment (required for assign action)",
    )
    response: Optional[str] = Field(
        None,
        description="Response message to the customer (required for respond action)",
    )

    @field_validator("action_type")
    @classmethod
    def validate_action_type(cls, v: str) -> str:
        allowed = {e.value for e in ActionType}
        if v not in allowed:
            raise ValueError(f"action_type must be one of {allowed}, got '{v}'")
        return v


class Reward(BaseModel):
    """Reward signal returned after each step."""
    score: float = Field(..., ge=0.0, le=1.0, description="Reward score between 0.0 and 1.0")
    feedback: str = Field(..., description="Human-readable explanation of the score")


class EnvState(BaseModel):
    """Full environment state for inspection."""
    observation: Observation
    step_count: int = Field(default=0, ge=0)
    done: bool = False
    episode_id: str = ""
    total_reward: float = 0.0
    task_name: str = ""
    max_steps: int = 5


class StepResult(BaseModel):
    """Return value of env.step()."""
    observation: Observation
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetResult(BaseModel):
    """Return value of env.reset()."""
    observation: Observation
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Task / Ground-Truth Models
# ---------------------------------------------------------------------------

class GroundTruth(BaseModel):
    """Expected correct answer for a ticket scenario."""
    issue_type: str = Field(..., description="Correct issue classification")
    team: Optional[str] = Field(None, description="Correct team assignment")
    action_type: str = Field(..., description="Correct action to take")
    response_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords that should appear in a correct response",
    )
    requires_refund: bool = False
    requires_escalation: bool = False


class TicketScenario(BaseModel):
    """A complete ticket scenario: the ticket itself plus the expected answer."""
    ticket: Observation
    ground_truth: GroundTruth
