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
            team="logistics_team",
            action_type="respond",
            response_keywords=["package", "tracking", "investigate", "delivered"],
        )
    ),
    TicketScenario(
        ticket=Observation(
            ticket_id="TKT-M002",
            issue_type="unknown",
            sentiment="angry",
            priority="critical",
            message="There is a dangerous defect with this product! It overheated and almost caught fire. You need to investigate this.",
        ),
        ground_truth=GroundTruth(
            issue_type="safety",
            team="safety_team",
            action_type="escalate",
            response_keywords=["safety", "investigation", "immediately", "hazard"],
            requires_escalation=True,
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
            team="tech_support_team",
            action_type="respond",
            response_keywords=["login", "password", "reset", "email"],
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
            team="finance_team",
            action_type="respond",
            response_keywords=["invoice", "company name", "corrected"],
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
            team="orders_team",
            action_type="respond",
            response_keywords=["bulk", "discount", "corporate", "units"],
        )
    ),
    TicketScenario(
        ticket=Observation(
            ticket_id="TKT-M006",
            issue_type="unknown",
            sentiment="neutral",
            priority="medium",
            message="Does your store ship internationally? I'm in Germany and want to know about shipping costs and delivery times.",
        ),
        ground_truth=GroundTruth(
            issue_type="shipping",
            team="logistics_team",
            action_type="respond",
            response_keywords=["international", "shipping", "Germany", "delivery"],
        )
    ),
    TicketScenario(
        ticket=Observation(
            ticket_id="TKT-M007",
            issue_type="unknown",
            sentiment="negative",
            priority="medium",
            message="My data in the app isn't syncing between my phone and computer. I'm losing work.",
        ),
        ground_truth=GroundTruth(
            issue_type="technical",
            team="tech_support_team",
            action_type="respond",
            response_keywords=["sync", "data", "device", "troubleshoot"],
        )
    ),
    TicketScenario(
        ticket=Observation(
            ticket_id="TKT-M008",
            issue_type="unknown",
            sentiment="negative",
            priority="high",
            message="I used a coupon code at checkout but was still charged the full price. This is a billing error.",
        ),
        ground_truth=GroundTruth(
            issue_type="billing",
            team="finance_team",
            action_type="refund",
            response_keywords=["coupon", "discount", "refund", "apologize"],
            requires_refund=True,
        )
    ),
    TicketScenario(
        ticket=Observation(
            ticket_id="TKT-M009",
            issue_type="unknown",
            sentiment="positive",
            priority="medium",
            message="Our company is growing fast. Can we upgrade to the Business plan and add 15 more user seats?",
        ),
        ground_truth=GroundTruth(
            issue_type="orders",
            team="orders_team",
            action_type="respond",
            response_keywords=["business plan", "upgrade", "seats", "users"],
        )
    ),
    TicketScenario(
        ticket=Observation(
            ticket_id="TKT-M010",
            issue_type="unknown",
            sentiment="angry",
            priority="critical",
            message="I just saw a news report that your product model XZ-400 has been recalled. I own one. What should I do?",
        ),
        ground_truth=GroundTruth(
            issue_type="safety",
            team="safety_team",
            action_type="escalate",
            response_keywords=["recall", "safety", "XZ-400", "immediately"],
            requires_escalation=True,
        )
    ),
]