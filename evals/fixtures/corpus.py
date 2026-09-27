"""§12.3: the synthetic corpus the framework owns, generated rather than hand-authored.

The plan says the owner is not required to author hundreds of fixtures, so this module is
the author: every record is produced from a template by a category that names the thing
§12.3 asks to be exercised — corrections, ambiguous dates, same-name people, quoted
duplicates, tail evidence in a long document, measurements with gaps, unknown answers, and
claims that cannot both be true.

Three rules hold the whole thing honest:

* Every record carries a `code` that appears in no other record. Recall is then a fact about
  retrieval, not about a scorer's tolerance for near misses, and abstention is checkable
  because an absent code has exactly one correct answer.
* Event time and arrival time are deliberately different, and some records have no event
  time at all. A corpus where those coincide cannot tell temporal correctness from luck.
* Nothing here is random. Two runs of the harness over the same tree must produce the same
  numbers, or a regression is indistinguishable from a dice roll.
"""
from __future__ import annotations

from typing import Any

# The categories §12.3 lists, in the order it lists them, with the count of records each
# contributes per person. The totals are asserted in the harness, not here: a corpus that
# quietly lost a category is the failure this file exists to make visible.
CATEGORIES = (
    "preference", "correction", "temporal-update", "ambiguous-date", "cross-source",
    "same-name", "unconfirmed-identity", "negation", "quoted-duplicate", "multilingual",
    "long-document-tail", "measurement", "sensor-gap", "attachment", "unknown-answer",
    "conflicting-facts", "procedure", "failure", "non-applicable",
)

PEOPLE = ("Amara", "Bruno", "Chidi", "Danica", "Elif", "Farid", "Greta", "Henok", "Ines",
          "Jonas", "Kira", "Lars")

# The questions nobody authored: an answer must exist in the corpus for these to be
# answerable, and every one of them has none. A tuple rather than a generator because the
# harness reads the corpus several times and an exhausted generator is quietly empty.
UNKNOWN_CODES = tuple(f"ZZ{i:04d}" for i in range(1, 121))

BASE = "2026-09"


def _envelope(code: str, *, person: str, source: str, category: str, text: str,
              occurred: str | None, precision: str = "second",
              observed: str = f"{BASE}-25T12:00:00+00:00", revision: str = "1",
              kind: str | None = None,
              attachments: list[dict[str, Any]] | None = None,
              measured: dict[str, Any] | None = None) -> dict[str, Any]:
    """One canonical envelope, with the code carried in the text and in the metadata.

    The code in the text is what a lexical search can find; the code in the metadata is what
    a scorer reads to know which record it was supposed to find. Nothing outside the
    canonical contract is added: an ingress that tolerates stray keys is one that tolerates
    them by accident.
    """
    metadata: dict[str, Any] = {"participants": [
        {"namespace": "email", "address": f"{person.lower()}@example.org",
         "display_name": person}], "code": code, "category": category}
    if attachments:
        metadata["attachments"] = attachments
    if measured:
        metadata["measured"] = measured
    return {
        "source": source,
        "source_id": f"{category}-{code}",
        "revision": revision,
        "kind": kind or ("email" if source == "gmail" else "message"),
        "text": f"{text} [ref {code}]",
        "observed_at": observed,
        "occurred_at": occurred,
        "occurred_precision": precision,
        "metadata": metadata,
    }


def _templates(person: str, index: int) -> list[tuple[str, dict[str, Any]]]:
    """Every category's records for one person.

    Each entry is (code-prefix-category, keyword arguments for `_envelope`). The strings are
    plain on purpose: a scorer compares identifiers, and prose that reads like a real message
    only matters for the model-backed rows, which this harness does not claim to measure.
    """
    lower = person.lower()
    tail = " ".join(f"line {n} of the weekly report for {lower}" for n in range(1, 121))
    return [
        ("preference", {
            "source": "gmail",
            "text": f"{person} prefers the afternoon slot for reviews",
            "occurred": f"{BASE}-0{index % 9 + 1}T09:12:00+01:00"}),
        ("correction", {
            "source": "gmail",
            "text": f"Correction: {person}'s flight is on the 14th, not the 12th",
            "occurred": f"{BASE}-11T18:02:00+01:00"}),
        ("temporal-update", {
            "source": "whatsapp",
            "text": f"The standup with {person} moved to Friday 07:30",
            "occurred": f"{BASE}-10T08:00:00+01:00"}),
        ("ambiguous-date", {
            "source": "gmail",
            "text": f"{person} mentioned the invoice from sometime in August",
            # A month-precision claim still needs a real instant to point at, so the day is
            # the first of the month and the precision field is what says the rest is a
            # guess. A consumer that reads this as "1 August, noon" is the bug this
            # record exists to catch.
            "occurred": "2026-08-01T00:00:00+00:00", "precision": "month"}),
        ("cross-source", {
            "source": "whatsapp",
            "text": f"{person} confirmed the lease figures from the email thread",
            "occurred": f"{BASE}-12T10:40:00+01:00"}),
        ("same-name", {
            "source": "gmail",
            "text": f"A different {person} (Sam {person}, not the one in the project) "
                    f"sent the budget note",
            "occurred": f"{BASE}-13T11:00:00+01:00"}),
        ("unconfirmed-identity", {
            "source": "whatsapp",
            "text": f"The number ending 44{index} that {person} used once is in the thread",
            "occurred": f"{BASE}-14T12:00:00+01:00"}),
        ("negation", {
            "source": "gmail",
            "text": f"{person} did not approve the cancellation and asked to keep it",
            "occurred": f"{BASE}-15T09:00:00+01:00"}),
        ("quoted-duplicate", {
            "source": "whatsapp",
            "text": f"FW: {person} wrote that the deposit was already paid",
            "occurred": f"{BASE}-16T09:30:00+01:00"}),
        ("multilingual", {
            "source": "whatsapp",
            "text": f"{person} а сказал что встреча перенесена — 会議は金曜日に移動しました",
            "occurred": f"{BASE}-17T09:45:00+01:00"}),
        ("long-document-tail", {
            "source": "files",
            "text": f"{tail}\nConclusion for {person}: the sensor threshold stays at 12",
            "occurred": f"{BASE}-18T07:00:00+01:00"}),
        ("measurement", {
            "source": "structured", "kind": "measurement",
            "text": f"{person} logged 7.4 km in 41:12",
            "occurred": f"{BASE}-19T06:30:00+01:00",
            "measured": {"value": "7.4", "unit": "km", "window": "daily"}}),
        ("sensor-gap", {
            "source": "structured", "kind": "measurement",
            "text": f"{person}'s sleep device reported nothing between 02:00 and 05:10",
            "occurred": None, "precision": "unknown",
            "measured": {"value": None, "unit": "minutes", "window": "gap"}}),
        ("attachment", {
            "source": "gmail",
            "text": f"{person} attached the signed lease scan",
            "occurred": f"{BASE}-20T15:00:00+01:00",
            "attachments": [{"name": "lease.pdf", "mime": "application/pdf",
                             "size": 20480, "text": "the lease ends in March"}]}),
        ("unknown-answer", {
            "source": "gmail",
            "text": f"{person} asked what the warranty on the drill is",
            "occurred": f"{BASE}-21T16:00:00+01:00"}),
        ("conflicting-facts", {
            "source": "whatsapp",
            "text": f"{person} says the meeting is at 10, and the calendar says 11",
            "occurred": f"{BASE}-22T17:00:00+01:00"}),
        ("procedure", {
            "source": "files",
            "text": f"How {person} renews the passport: form, photo, then the office",
            "occurred": f"{BASE}-23T08:00:00+01:00"}),
        ("failure", {
            "source": "gmail",
            "text": f"The export {person} tried failed twice with a checksum mismatch",
            "occurred": f"{BASE}-24T09:00:00+01:00"}),
        ("non-applicable", {
            "source": "whatsapp",
            "text": f"{person} asked about a car service while living abroad",
            "occurred": f"{BASE}-25T10:00:00+01:00"}),
    ]


def records() -> list[dict[str, Any]]:
    """The corpus: every category, every person, one distinct code per record.

    Two corrections are deliberately written as *revisions of one source item* rather than as
    two messages, because a store that cannot tell a revision from a duplicate has no way to
    answer "what is true now" — and that is the property the category is here to test.
    """
    found: list[dict[str, Any]] = []
    for index, person in enumerate(PEOPLE):
        for category, arguments in _templates(person, index):
            code = f"{category[:2].upper()}{index:02d}{_ordinal(category)}"
            found.append(_envelope(code, person=person, category=category, **arguments))
        # The revision pair, keyed on the same (source, source_id) with new bytes.
        first = _envelope(f"CO{index:02d}R1", person=person, category="correction",
                          source="gmail", text=f"{person}'s flight is on the 12th",
                          occurred=f"{BASE}-09T18:00:00+01:00")
        second = _envelope(f"CO{index:02d}R2", person=person, category="correction",
                           source="gmail", revision="2",
                           text=f"Correction: {person}'s flight is on the 14th",
                           occurred=f"{BASE}-11T18:02:00+01:00")
        # A revision is the *same source item* with new bytes, and (source, source_id) is
        # what names the item. Inheriting the first envelope's id rather than deriving one
        # from the new code is the whole difference between a correction and a duplicate.
        second["source_id"] = first["source_id"]
        found.extend((first, second))
    return found


def _ordinal(category: str) -> str:
    """A stable two-character suffix per category, so codes never collide."""
    return f"{CATEGORIES.index(category):02d}"


def questions() -> list[dict[str, Any]]:
    """Held-out questions: one per record, plus the ones that have no answer.

    The gold for a retrieval question is the code the record carries, so the scorer asks a
    yes-or-no question of the packet rather than grading a paraphrase. Three answers are
    possible, and conflating the last two is how a corrected item is scored as a miss:

    * `retrieve` — the answer is in the archive and must come back.
    * `abstain` — the answer is nowhere in the corpus, so nothing must come back.
    * `superseded` — the text exists but was corrected, so *that* text must not come back and
      its replacement must.
    """
    withdrawn = {f"CO{index:02d}R1" for index in range(len(PEOPLE))}
    asked = [{"query": f"ref {item['metadata']['code']}", "gold": item["metadata"]["code"],
              "expect": "superseded" if item["metadata"]["code"] in withdrawn else "retrieve",
              "category": item["metadata"]["category"], "source": item["source"]}
             for item in records()]
    asked += [{"query": f"ref {code}", "gold": code, "expect": "abstain",
               "category": "unknown-answer", "source": "none"} for code in UNKNOWN_CODES]
    return asked


def conflicts() -> list[dict[str, Any]]:
    """Pairs that cannot both be true, for the contradiction row.

    Written as assertions the store accepts rather than as prose to be judged: the claim
    under test is that a clash is *found and named*, which is a deterministic property.
    """
    return [{"subject": f"person:{person.lower()}-meeting", "predicate": "start-time",
             "values": ("10:00", "11:00"), "category": "conflicting-facts"}
            for person in PEOPLE]


def scenarios() -> list[dict[str, Any]]:
    """100 moments worth interrupting for, and 100 matched ones that are not.

    Each pair differs in exactly one structural fact the gate reads — coverage, staleness,
    an opt-out, quiet hours, a spent budget, a duplicate intention, a disconnected source —
    so a pass measures the gate rather than the luck of the wording.
    """
    eligible, no_action = [], []
    reasons = ("opt-out", "quiet hours", "stale evidence", "backfill coverage",
               "disconnected source", "duplicate intention", "spent budget",
               "paused topic", "shadow mode", "unknown coverage")
    for index in range(100):
        person = PEOPLE[index % len(PEOPLE)]
        hour = 9 + (index % 8)
        moment = f"{BASE}-{(index % 26) + 1:02d}T{hour:02d}:15:00+01:00"
        code = f"EL{index:04d}"
        eligible.append({"case": "eligible", "code": code, "person": person,
                         "topic": "general", "at": moment, "source": "gmail"})
        reason = reasons[index % len(reasons)]
        no_action.append({"case": "no-action", "reason": reason, "code": f"NA{index:04d}",
                          "person": person,
                          "topic": f"topic-{index % 5}" if reason in
                          ("opt-out", "paused topic") else "general",
                          "at": _quiet_moment(moment) if reason == "quiet hours" else moment,
                          "source": "gmail" if reason != "disconnected source" else "off",
                          "index": index})
    return eligible + no_action


def _quiet_moment(moment: str) -> str:
    """The same day, inside the default quiet window — stated in UTC so the window is
    unambiguous: 22:30 at an offset of +01:00 is 21:30 Zulu, which is nobody's quiet hour.
    """
    return f"{moment[:11]}22:30:00+00:00"


def totals() -> dict[str, int]:
    """What the corpus claims to hold, so a category that vanished is not silently missing.

    `codes` is counted separately from `records` because the two only agree if every code is
    unique — which is the assumption every scorer in `checks.py` rests on. A duplicated code
    would make recall a coin flip and abstention unanswerable.
    """
    corpus = records()
    by_category: dict[str, int] = {}
    for item in corpus:
        key = str(item["metadata"]["category"])
        by_category[key] = by_category.get(key, 0) + 1
    asked = questions()
    scenes = scenarios()
    return {"records": len(corpus),
            "codes": len({str(item["metadata"]["code"]) for item in corpus}),
            "categories": len(by_category),
            "people": len(PEOPLE),
            "questions": len(asked),
            "answerable": sum(1 for row in asked if row["expect"] == "retrieve"),
            "abstentions": sum(1 for row in asked if row["expect"] == "abstain"),
            "superseded": sum(1 for row in asked if row["expect"] == "superseded"),
            "conflicts": len(conflicts()),
            "eligible_scenarios": sum(1 for row in scenes if row["case"] == "eligible"),
            "no_action_scenarios": sum(1 for row in scenes if row["case"] == "no-action"),
            **{f"record_{key}": value for key, value in sorted(by_category.items())}}
