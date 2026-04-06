from openenv.models import TicketScenario, Observation, GroundTruth

HARD_TASKS = [
    TicketScenario(
        ticket=Observation(
            ticket_id="TKT-H001",
            issue_type="unknown",
            sentiment="angry",
            priority="critical",
            message="I've been waiting for my refund for 3 weeks! I spoke to someone last week who said it was processed. Let me speak to your manager right now! Order #12345.",
        ),
        ground_truth=GroundTruth(
            issue_type="complaint",
            action_type="escalate",
            team="management_team",
            response_keywords=["apologize", "manager", "escalated", "order #12345"],
            requires_escalation=True
        )
    ),
    TicketScenario(
        ticket=Observation(
            ticket_id="TKT-H002",
            issue_type="unknown",
            sentiment="negative",
            priority="high",
            message="I ordered the blue shirt but received a red one. I want a full refund and I'm not paying return shipping for your mistake.",
        ),
        ground_truth=GroundTruth(
            issue_type="returns",
            action_type="refund",
            response_keywords=["apologize", "refund", "return shipping", "red"],
            requires_refund=True
        )
    ),
    TicketScenario(
        ticket=Observation(
            ticket_id="TKT-H003",
            issue_type="unknown",
            sentiment="neutral",
            priority="medium",
            message="Can you explain how the API rate limits work? Do they reset hourly or daily? I keep getting 429 errors.",
        ),
        ground_truth=GroundTruth(
            issue_type="technical",
            action_type="respond",
            response_keywords=["api", "rate limit", "hourly", "daily", "429"]
        )
    ),
    TicketScenario(
        ticket=Observation(
            ticket_id="TKT-H004",
            issue_type="unknown",
            sentiment="positive",
            priority="low",
            message="Thanks for the great service last time! I have a new question: does the standard plan include custom domains?",
        ),
        ground_truth=GroundTruth(
            issue_type="sales",
            action_type="respond",
            response_keywords=["standard plan", "custom domains"]
        )
    ),
    TicketScenario(
        ticket=Observation(
            ticket_id="TKT-H005",
            issue_type="unknown",
            sentiment="angry",
            priority="critical",
            message="Your software deleted all my customer data!!! I need someone to recover it IMMEDIATELY.",
        ),
        ground_truth=GroundTruth(
            issue_type="technical",
            action_type="escalate",
            team="tech_support_team",
            response_keywords=["data", "recover", "urgently", "escalated"],
            requires_escalation=True
        )
    )
]
