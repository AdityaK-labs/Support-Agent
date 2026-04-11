from openenv.models import TicketScenario, Observation, GroundTruth

EASY_TASKS = [
    TicketScenario(
        ticket=Observation(
            ticket_id="TKT-E001",
            issue_type="unknown",
            sentiment="neutral",
            priority="low",
            message="Hi, I just placed an order but I put the wrong shipping address. Can I change it?",
        ),
        ground_truth=GroundTruth(
            issue_type="shipping",
            team="logistics_team",
            action_type="respond",
            response_keywords=["shipping address", "updated", "order"],
        )
    ),
    TicketScenario(
        ticket=Observation(
            ticket_id="TKT-E002",
            issue_type="unknown",
            sentiment="negative",
            priority="medium",
            message="My credit card was charged twice for the same purchase! Please help.",
        ),
        ground_truth=GroundTruth(
            issue_type="billing",
            team="finance_team",
            action_type="refund",
            response_keywords=["double charge", "refund", "apologize"],
            requires_refund=True,
        )
    ),
    TicketScenario(
        ticket=Observation(
            ticket_id="TKT-E003",
            issue_type="unknown",
            sentiment="positive",
            priority="low",
            message="I love your product, but I'm trying to figure out how to update my profile picture.",
        ),
        ground_truth=GroundTruth(
            issue_type="technical",
            team="tech_support_team",
            action_type="respond",
            response_keywords=["profile", "settings", "update"],
        )
    ),
    TicketScenario(
        ticket=Observation(
            ticket_id="TKT-E004",
            issue_type="unknown",
            sentiment="angry",
            priority="high",
            message="This item arrived completely broken. I want to return it immediately.",
        ),
        ground_truth=GroundTruth(
            issue_type="returns",
            team="orders_team",
            action_type="refund",
            response_keywords=["broken", "refund", "apologize", "return"],
            requires_refund=True,
        )
    ),
    TicketScenario(
        ticket=Observation(
            ticket_id="TKT-E005",
            issue_type="unknown",
            sentiment="neutral",
            priority="medium",
            message="I need to cancel my subscription renewal before next month.",
        ),
        ground_truth=GroundTruth(
            issue_type="cancellation",
            team="orders_team",
            action_type="respond",
            response_keywords=["subscription", "cancel", "renewal"],
        )
    )
]