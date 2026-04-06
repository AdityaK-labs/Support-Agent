from typing import Dict, Any
from openenv.models import Action, GroundTruth, Reward

class GraderEngine:
    @staticmethod
    def grade(action: Action, ground_truth: GroundTruth, step_count: int) -> Reward:
        score = 0.0
        feedback_parts = []
        
        # 1. Classification (max 0.2)
        if action.action_type in ["classify", "assign", "respond", "refund", "escalate"]:
            # In our setup, classification happens inherently if you pick the correct action type and context.
            pass # We'll assess action_type correctly instead.
            
        # For simplicity, if action type matches entirely:
        if action.action_type == ground_truth.action_type:
            score += 0.3
            feedback_parts.append(f"Correct action type ({action.action_type}): +0.3")
        else:
            feedback_parts.append(f"Incorrect action type (Expected {ground_truth.action_type}, got {action.action_type})")
            
        # 2. Team Assignment (max 0.3)
        if ground_truth.team:
            if action.team == ground_truth.team:
                score += 0.3
                feedback_parts.append(f"Correct team assignment ({action.team}): +0.3")
            elif action.team:
                feedback_parts.append(f"Wrong team assignment (Expected {ground_truth.team}, got {action.team})")
            else:
                feedback_parts.append("Missing team assignment.")
                
        # 3. Response Quality (max 0.4)
        if ground_truth.response_keywords:
            if action.response:
                matched = [kw for kw in ground_truth.response_keywords if kw.lower() in action.response.lower()]
                kw_score = min(0.4, (len(matched) / len(ground_truth.response_keywords)) * 0.4)
                score += kw_score
                feedback_parts.append(f"Response keywords matched ({len(matched)}/{len(ground_truth.response_keywords)}): +{kw_score:.2f}")
            else:
                feedback_parts.append("No response provided.")
                
        # Penalties
        if action.action_type == "refund" and not ground_truth.requires_refund:
            score -= 0.5
            feedback_parts.append("Penalty: Unnecessary refund (-0.5)")
            
        if action.action_type == "escalate" and not ground_truth.requires_escalation:
            score -= 0.3
            feedback_parts.append("Penalty: Unnecessary escalation (-0.3)")
            
        # Exessive steps penalty
        if step_count > 3:
            penalty = (step_count - 3) * 0.1
            score -= penalty
            feedback_parts.append(f"Penalty: Excessive steps (-{penalty:.2f})")
            
        final_score = max(0.0, min(1.0, score))
        if final_score == 0.0 and score < 0:
            feedback_parts.append("Score clipped to baseline 0.0.")
            
        # Base credit for trying
        if final_score == 0.0:
            final_score = 0.1
            feedback_parts.append("Base credit: +0.1")
            
        return Reward(score=final_score, feedback="; ".join(feedback_parts))
