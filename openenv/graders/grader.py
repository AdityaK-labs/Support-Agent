from typing import Dict, Any
from openenv.models import Action, GroundTruth, Reward

class GraderEngine:
    @staticmethod
    def grade(action: Action, ground_truth: GroundTruth, step_count: int) -> Reward:
        """
        Normalized grader: score is proportional to what the scenario demands.
        
        Scoring breakdown (before normalization):
          +0.3  correct action_type
          +0.3  correct team assignment (only if ground_truth.team exists)
          +0.4  response keyword coverage (only if ground_truth.response_keywords exist)
        
        Normalization: final positives are divided by max_possible so that
        a scenario that only tests action_type (e.g. Easy tasks = classify)
        can legitimately reach 1.0.

        Punishments (applied after normalization, before clamping):
          -0.50  Unnecessary refund issued
          -0.30  Unnecessary escalation
          -0.15  Wrong team selected (when ground_truth has a team)
          -0.10  per step beyond 3 (excessive steps)
        
        Final score is clamped to [0.0, 1.0].
        """
        raw_score = 0.0
        max_possible = 0.0
        feedback_parts = []
        punishment_total = 0.0

        # ------------------------------------------------------------------ #
        # 1. Action Type Classification (max contribution: 0.3)
        # ------------------------------------------------------------------ #
        if ground_truth.action_type:
            max_possible += 0.3
            if action.action_type == ground_truth.action_type:
                raw_score += 0.3
                feedback_parts.append(f"[+] Correct action_type='{action.action_type}': +0.30")
            else:
                feedback_parts.append(
                    f"[-] Wrong action_type: expected='{ground_truth.action_type}', "
                    f"got='{action.action_type}'"
                )

        # ------------------------------------------------------------------ #
        # 2. Team Assignment (max contribution: 0.3, only if gt.team set)
        # ------------------------------------------------------------------ #
        if ground_truth.team:
            max_possible += 0.3
            if action.team == ground_truth.team:
                raw_score += 0.3
                feedback_parts.append(f"[+] Correct team='{action.team}': +0.30")
            elif action.team:
                # Picked SOMETHING but wrong — half-credit removed as punishment
                punishment_total += 0.15
                feedback_parts.append(
                    f"[-] Wrong team: expected='{ground_truth.team}', "
                    f"got='{action.team}' | Punishment: -0.15"
                )
            else:
                feedback_parts.append(f"[-] Missing team (expected '{ground_truth.team}')")

        # ------------------------------------------------------------------ #
        # 3. Response Keyword Coverage (max contribution: 0.4, only if gt.keywords set)
        # ------------------------------------------------------------------ #
        if ground_truth.response_keywords:
            max_possible += 0.4
            if action.response:
                matched = [
                    kw for kw in ground_truth.response_keywords
                    if kw.lower() in action.response.lower()
                ]
                coverage = len(matched) / len(ground_truth.response_keywords)
                kw_score = min(0.4, coverage * 0.4)
                raw_score += kw_score
                feedback_parts.append(
                    f"[+] Keywords matched {len(matched)}/{len(ground_truth.response_keywords)} "
                    f"({coverage*100:.0f}% coverage): +{kw_score:.2f}"
                )
                if matched:
                    feedback_parts.append(f"    Matched: {matched}")
                missing = [kw for kw in ground_truth.response_keywords if kw.lower() not in action.response.lower()]
                if missing:
                    feedback_parts.append(f"    Missing: {missing}")
            else:
                feedback_parts.append("[-] No response text provided (required for keyword score)")

        # ------------------------------------------------------------------ #
        # Normalize raw_score → proportional_score ∈ [0.0, 1.0]
        # ------------------------------------------------------------------ #
        proportional_score = (raw_score / max_possible) if max_possible > 0 else 0.0
        feedback_parts.append(
            f"[=] Proportional score: {raw_score:.2f}/{max_possible:.2f} = {proportional_score:.2f}"
        )

        # ------------------------------------------------------------------ #
        # Punishments (applied after normalization, on top of proportional score)
        # ------------------------------------------------------------------ #
        # Unnecessary refund — heavy punishment
        if action.action_type == "refund" and not ground_truth.requires_refund:
            punishment_total += 0.50
            feedback_parts.append("[!] PUNISHMENT: Unnecessary refund (-0.50)")

        # Unnecessary escalation — moderate punishment
        if action.action_type == "escalate" and not ground_truth.requires_escalation:
            punishment_total += 0.30
            feedback_parts.append("[!] PUNISHMENT: Unnecessary escalation (-0.30)")

        # Excessive steps — creeping penalty
        if step_count > 3:
            step_penalty = (step_count - 3) * 0.10
            punishment_total += step_penalty
            feedback_parts.append(f"[!] PUNISHMENT: Excessive steps (step {step_count}) (-{step_penalty:.2f})")

        if punishment_total > 0:
            feedback_parts.append(f"[=] Total punishment: -{punishment_total:.2f}")

        # ------------------------------------------------------------------ #
        # Final score: clamp to [0.0, 1.0]
        # ------------------------------------------------------------------ #
        final_score = max(0.002, min(0.998, proportional_score - punishment_total))
        feedback_parts.append(f"[FINAL] Score: {final_score:.3f}")

        return Reward(score=final_score, feedback=" | ".join(feedback_parts))
