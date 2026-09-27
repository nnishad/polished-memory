"""The lesson case set that more than one suite has to answer to.

``RULE``, ``CASES`` and ``propose`` were written inside the learning suite and then
needed again by the evaluation door and by the explanation of a verdict. Shared case
data lives here rather than in a test module on purpose: if the door and the rules
each hold their own copy, a change to one of them stops testing the same lesson, and
the two suites go on agreeing with themselves.
"""
from __future__ import annotations

AGENT = "hermes-agent"
RULE = {"all": [{"field": "source", "op": "eq", "value": "gmail"}]}
INVOICE_RULE = {"all": [{"field": "source", "op": "eq", "value": "gmail"},
                        {"field": "topic", "op": "any", "value": ["billing", "invoice"]}]}
CASES = ({"case_id": "overdue-invoice", "role": "targeted", "task": {"source": "gmail"}},
         {"case_id": "meeting-moved", "role": "regression", "task": {"source": "gmail"}})


class Runner:
    """Separately authorised evaluation machinery, or an imitation of it."""

    name = "fixture-suite"

    def __init__(self, results=None, default="pass"):
        self.results = results or {}
        self.default = default
        self.asked: list[str] = []

    def evaluate(self, *, lesson, case):
        self.asked.append(str(case["case_id"]))
        outcome = self.results.get(str(case["case_id"]), self.default)
        if isinstance(outcome, dict):
            return dict(outcome)
        return {"outcome": outcome, "detail": f"runner said {outcome}"}


def propose(lessons, cite, lesson_id="chase-invoice", **kwargs):
    params = {"evidence": [cite] if cite else [],
              "text": "Chase an overdue invoice twice by mail before phoning.",
              "applicability": RULE,
              "proposed_by": AGENT, "proposed_kind": "agent"}
    params.update(kwargs)
    return lessons.propose(lesson_id=lesson_id, **params)
