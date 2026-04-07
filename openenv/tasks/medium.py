from openenv.models import TicketScenario, Observation, GroundTruth

MEDIUM_TASKS = [
    TicketScenario(
        ticket=Observation(
            ticket_id="TKT-M001",
            issue_type="unknown",
            sentiment="neutral",
            priority="medium",
            message="Where is my package? The tracking says delivered but I haven't received anything.",
        ),
        ground_truth=GroundTruth(
            issue_type="shipping",
            action_type="assign",
            team="logistics_team"
        )
    ),
    TicketScenario(
        ticket=Observation(
            ticket_id="TKT-M002",
            issue_type="unknown",
            sentiment="angry",
            priority="high",
            message="There is a dangerous defect with this product! It overheated and almost caught fire. You need to investigate this.",
        ),
        ground_truth=GroundTruth(
            issue_type="safety",
            action_type="assign",
            team="safety_team",
            requires_escalation=True
        )
    ),
    TicketScenario(
        ticket=Observation(
            ticket_id="TKT-M003",
            issue_type="unknown",
            sentiment="negative",
            priority="medium",
            message="I can't log into my account. The password reset link never arrives in my email.",
        ),
        ground_truth=GroundTruth(
            issue_type="technical",
            action_type="assign",
            team="tech_support_team"
        )
    ),
    TicketScenario(
        ticket=Observation(
            ticket_id="TKT-M004",
            issue_type="unknown",
            sentiment="neutral",
            priority="high",
            message="We received an invoice but the company name is spelled wrong. We need a revised invoice.",
        ),
        ground_truth=GroundTruth(
            issue_type="billing",
            action_type="assign",
            team="finance_team"
        )
    ),
    TicketScenario(
        ticket=Observation(
            ticket_id="TKT-M005",
            issue_type="unknown",
            sentiment="neutral",
            priority="low",
            message="I want to know if you offer bulk discounts for corporate orders over 100 units.",
        ),
        ground_truth=GroundTruth(
            issue_type="orders",
            action_type="assign",
            team="orders_team"
        )
    )
]
