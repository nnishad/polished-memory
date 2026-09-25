"""C11 evaluated procedural learning: what worked, and what could prove it.

Three authorities that must not collapse into one. The outcome log records what
happened, from whom, and what that is evidence *of*. The lesson store holds versioned
rules the system can actually evaluate. The evaluation ledger is the only thing that
may turn a candidate into an active lesson, and it does so from cases it watched, not
from a verdict somebody typed in.
"""
from .evaluation import Evaluation, EvaluationLedger
from .lessons import Lesson, LessonStore
from .outcomes import Outcome, OutcomeLog

__all__ = ["OutcomeLog", "Outcome", "LessonStore", "Lesson", "EvaluationLedger",
           "Evaluation"]
