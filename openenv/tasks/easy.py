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
    ),
    TicketScenario(
        ticket=Observation(
            ticket_id="TKT-E006",
            issue_type="unknown",
            sentiment="negative",
            priority="medium",
            message="I ordered a blue laptop bag but received a black one instead. This is not what I paid for.",
        ),
        ground_truth=GroundTruth(
            issue_type="returns",
            team="orders_team",
            action_type="refund",
            response_keywords=["wrong item", "refund", "apologize", "return"],
            requires_refund=True,
        )
    ),
    TicketScenario(
        ticket=Observation(
            ticket_id="TKT-E007",
            issue_type="unknown",
            sentiment="negative",
            priority="high",
            message="I've been locked out of my account after enabling two-factor authentication. The codes aren't working.",
        ),
        ground_truth=GroundTruth(
            issue_type="technical",
            team="tech_support_team",
            action_type="respond",
            response_keywords=["two-factor", "account", "locked", "access"],
        )
    ),
    TicketScenario(
        ticket=Observation(
            ticket_id="TKT-E008",
            issue_type="unknown",
            sentiment="positive",
            priority="low",
            message="I'm placing a gift order — is it possible to add a gift wrap and a personal message card?",
        ),
        ground_truth=GroundTruth(
            issue_type="orders",
            team="orders_team",
            action_type="respond",
            response_keywords=["gift", "wrap", "message", "order"],
        )
    ),
    TicketScenario(
        ticket=Observation(
            ticket_id="TKT-E009",
            issue_type="unknown",
            sentiment="neutral",
            priority="low",
            message="I saw the same product at a competitor for $15 cheaper. Do you offer price matching?",
        ),
        ground_truth=GroundTruth(
            issue_type="billing",
            team="finance_team",
            action_type="respond",
            response_keywords=["price match", "competitor", "policy"],
        )
    ),
    TicketScenario(
        ticket=Observation(
            ticket_id="TKT-E010",
            issue_type="unknown",
            sentiment="neutral",
            priority="low",
            message="I recently moved and need to update my billing address on file. How do I do that?",
        ),
        ground_truth=GroundTruth(
            issue_type="billing",
            team="finance_team",
            action_type="respond",
            response_keywords=["billing address", "update", "account"],
        )
    ),
]