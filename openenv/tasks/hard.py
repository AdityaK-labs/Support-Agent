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
            team="management_team",
            action_type="escalate",
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
            team="orders_team",
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
            team="tech_support_team",
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
            team="orders_team",
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
            team="tech_support_team",
            action_type="escalate",
            response_keywords=["data", "recover", "urgently", "escalated"],
            requires_escalation=True
        )
    ),
    TicketScenario(
        ticket=Observation(
            ticket_id="TKT-H006",
            issue_type="unknown",
            sentiment="angry",
            priority="critical",
            message="Someone accessed my account without my permission and made purchases. I think I've been hacked. This is a security emergency.",
        ),
        ground_truth=GroundTruth(
            issue_type="technical",
            team="tech_support_team",
            action_type="escalate",
            response_keywords=["security", "breach", "account", "immediately", "escalated"],
            requires_escalation=True
        )
    ),
    TicketScenario(
        ticket=Observation(
            ticket_id="TKT-H007",
            issue_type="unknown",
            sentiment="angry",
            priority="critical",
            message="The battery in your device swelled up and cracked my laptop screen! It could have started a fire. I demand immediate action.",
        ),
        ground_truth=GroundTruth(
            issue_type="safety",
            team="safety_team",
            action_type="escalate",
            response_keywords=["battery", "defect", "safety", "immediately", "escalated"],
            requires_escalation=True
        )
    ),
    TicketScenario(
        ticket=Observation(
            ticket_id="TKT-H008",
            issue_type="unknown",
            sentiment="negative",
            priority="medium",
            message="I never agreed to auto-renewal on my annual plan. You charged me without clear consent. I want a full refund for this unauthorized charge.",
        ),
        ground_truth=GroundTruth(
            issue_type="billing",
            team="finance_team",
            action_type="respond",
            response_keywords=["auto-renewal", "policy", "refund", "subscription"],
        )
    ),
    TicketScenario(
        ticket=Observation(
            ticket_id="TKT-H009",
            issue_type="unknown",
            sentiment="angry",
            priority="critical",
            message="My account was disabled without any notice! I have 3 active enterprise projects and my whole team is locked out. I need this escalated NOW.",
        ),
        ground_truth=GroundTruth(
            issue_type="technical",
            team="management_team",
            action_type="escalate",
            response_keywords=["account", "disabled", "enterprise", "escalated", "immediately"],
            requires_escalation=True
        )
    ),
    TicketScenario(
        ticket=Observation(
            ticket_id="TKT-H010",
            issue_type="unknown",
            sentiment="negative",
            priority="high",
            message="I was charged the full price during your 40% off promotional event. The discount was supposed to apply automatically. Fix this.",
        ),
        ground_truth=GroundTruth(
            issue_type="billing",
            team="finance_team",
            action_type="refund",
            response_keywords=["promotion", "discount", "refund", "apologize"],
            requires_refund=True
        )
    ),
]