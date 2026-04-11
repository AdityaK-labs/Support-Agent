from typing import Dict, Any
from openenv.models import Action, GroundTruth, Reward


class GraderEngine:

    @staticmethod
    def grade_phase(action: Action, ground_truth: GroundTruth, phase: int, step_count: int) -> Reward:
        """
        Phase-aware grading for 3-phase RL episodes.

        Phase 1 — Triage:
            Agent must classify the ticket.
            Score: 0.998 if action_type=="classify", else 0.002.

        Phase 2 — Route:
            Agent must assign to the correct team.
            Score: 0.998 for correct team, 0.4 for wrong team, 0.002 for wrong action type.

        Phase 3 — Resolve:
            Agent must take the correct resolution action (respond/refund/escalate)
            with sufficient response quality.
            Score: full GraderEngine.grade() output (0.002 – 0.998).

        Per-step rewards are averaged across phases in inference.py.
        A perfect 3-step episode averages to ~0.998.
        """
        if phase == 1:
            if action.action_type == "classify":
                return Reward(
                    score=0.998,
                    feedback="[+] Phase 1 (Triage): Correct classify action | +0.998"
                )
            else:
                return Reward(
                    score=0.002,
                    feedback=f"[-] Phase 1 (Triage): Expected classify, got '{action.action_type}' | 0.002"
                )

        elif phase == 2:
            if action.action_type != "assign":
                return Reward(
                    score=0.002,
                    feedback=f"[-] Phase 2 (Route): Expected assign, got '{action.action_type}' | 0.002"
                )
            if not ground_truth.team:
                return Reward(
                    score=0.998,
                    feedback="[+] Phase 2 (Route): Assign accepted (no team constraint) | +0.998"
                )
            if action.team == ground_truth.team:
                return Reward(
                    score=0.998,
                    feedback=f"[+] Phase 2 (Route): Correct team '{action.team}' | +0.998"
                )
            elif action.team:
                return Reward(
                    score=0.4,
                    feedback=f"[-] Phase 2 (Route): Wrong team — expected '{ground_truth.team}', got '{action.team}' | 0.4"
                )
            else:
                return Reward(
                    score=0.002,
                    feedback=f"[-] Phase 2 (Route): Missing team (expected '{ground_truth.team}') | 0.002"
                )

        elif phase == 3:
            return GraderEngine.grade(action, ground_truth, step_count)

        else:
            return Reward(score=0.002, feedback=f"[!] Unknown phase {phase}")

    @staticmethod
    def grade(action: Action, ground_truth: GroundTruth, step_count: int) -> Reward:
        """
        Normalized grader for the resolution phase (phase 3).

        Scoring breakdown (before normalization):
          +0.3  correct action_type
          +0.3  correct team assignment (only if ground_truth.team exists)
          +0.4  response keyword coverage (only if ground_truth.response_keywords exist)

        Normalization: final positives are divided by max_possible so that
        a scenario that only tests action_type can legitimately reach 1.0.

        Punishments (applied after normalization, before clamping):
          -0.50  Unnecessary refund issued
          -0.30  Unnecessary escalation
          -0.15  Wrong team selected (when ground_truth has a team)
          -0.10  per step beyond 3 (excessive steps)

        Final score is clamped to [0.002, 0.998].
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
                punishment_total += 0.15
                feedback_parts.append(
                    f"[-] Wrong team: expected='{ground_truth.team}', "
                    f"got='{action.team}' | Punishment: -0.15"
                )
            else:
                feedback_parts.append(f"[-] Missing team (expected '{ground_truth.team}')")

        # ------------------------------------------------------------------ #
        # 3. Response Keyword Coverage (max contribution: 0.4)
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
        # Normalize
        # ------------------------------------------------------------------ #
        proportional_score = (raw_score / max_possible) if max_possible > 0 else 0.0
        feedback_parts.append(
            f"[=] Proportional score: {raw_score:.2f}/{max_possible:.2f} = {proportional_score:.2f}"
        )

        # ------------------------------------------------------------------ #
        # Punishments
        # ------------------------------------------------------------------ #
        if action.action_type == "refund" and not ground_truth.requires_refund:
            punishment_total += 0.50
            feedback_parts.append("[!] PUNISHMENT: Unnecessary refund (-0.50)")

        if action.action_type == "escalate" and not ground_truth.requires_escalation:
            punishment_total += 0.30
            feedback_parts.append("[!] PUNISHMENT: Unnecessary escalation (-0.30)")

        if step_count > 3:
            step_penalty = (step_count - 3) * 0.10
            punishment_total += step_penalty
            feedback_parts.append(f"[!] PUNISHMENT: Excessive steps (step {step_count}) (-{step_penalty:.2f})")

        if punishment_total > 0:
            feedback_parts.append(f"[=] Total punishment: -{punishment_total:.2f}")

        final_score = max(0.002, min(0.998, proportional_score - punishment_total))
        feedback_parts.append(f"[FINAL] Score: {final_score:.3f}")

        return Reward(score=final_score, feedback=" | ".join(feedback_parts))