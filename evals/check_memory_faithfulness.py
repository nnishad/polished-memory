"""Synthetic-only support/contradiction tests through the configured admission gate.

No personal archive access, remote bank writes, runtime changes or message delivery.
The small human-authored fixture is a smoke gate, not multilingual calibration.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from hermes_memory.backend.hindsight_client import HindsightError
from hermes_memory.config import load_settings
from hermes_memory.processing.faithfulness import MAX_CLAIMS
from hermes_memory.processing.summarization import client_for

SOURCES = {
    "allergy": "Nisha is allergic to peanuts.",
    "hinglish": "Arjun: Mujhe abhi chai nahi chahiye, coffee pasand hai.",
    "hindi": "मीरा: शायद शुक्रवार को जयपुर जाऊँगी। टिकट अभी बुक नहीं किया है।",
    "conditional": "Ravi: Agar baarish hui toh main Pune nahi jaunga.",
    "reported": "Dev: Karan ne kaha ki usko almonds pasand hain. Mere baare mein nahi kaha.",
    "english": "Asha: I might take a train tomorrow. I have not booked a ticket.",
    "heldout": "Riya: Shayad Monday ko Delhi jaungi; abhi ticket nahi liya. "
               "Mohan ne kaha usko doodh se allergy hai, par severity nahi batayi.",
}

# (source key, candidate claim, should be accepted). Unsupported vs contradicted
# labels are recorded, but any rejection is acceptable for the non-supported gold.
CASES = (
    ("allergy", "Nisha is allergic to peanuts.", True),
    ("allergy", "Nisha weighs 90 kg.", False),
    ("allergy", "Nisha has a severe peanut allergy.", False),
    ("allergy", "Ravi is allergic to peanuts.", False),
    ("allergy", "Nisha is not allergic to peanuts.", False),
    ("allergy", "Nisha is allergic to peanuts and weighs 90 kg.", False),
    ("allergy", "Nisha must avoid every kind of nut.", False),
    ("allergy", "Nisha ko peanuts se allergy hai.", True),
    ("allergy", "निशा को मूँगफली से एलर्जी है।", True),
    ("hinglish", "Arjun does not want tea right now and likes coffee.", True),
    ("hinglish", "Arjun has always hated tea.", False),
    ("hinglish", "Arjun abhi chai chahta hai.", False),
    ("hinglish", "Ravi likes coffee.", False),
    ("hinglish", "अर्जुन को कॉफी पसंद है।", True),
    ("hinglish", "Arjun prefers coffee because tea makes him ill.", False),
    ("hindi", "Meera might go to Jaipur on Friday and has not booked a ticket yet.", True),
    ("hindi", "Meera has booked a train to Jaipur for Friday at 9 am.", False),
    ("hindi", "Meera pakka Friday ko Jaipur jayegi.", False),
    ("hindi", "मीरा ने टिकट बुक कर लिया है।", False),
    ("hindi", "मीरा शायद शुक्रवार को जयपुर जाएगी।", True),
    ("conditional", "Ravi says he will not go to Pune if it rains.", True),
    ("conditional", "Ravi has cancelled his Pune trip.", False),
    ("conditional", "It will rain and Ravi will stay home.", False),
    ("conditional", "Agar baarish hui toh Ravi Pune nahi jayega.", True),
    ("reported", "Dev reports that Karan said he likes almonds.", True),
    ("reported", "Dev likes almonds.", False),
    ("reported", "Karan is allergic to almonds.", False),
    ("english", "Asha might take a train tomorrow.", True),
    ("english", "Asha has booked a train for tomorrow.", False),
    ("english", "Asha kal train se pakka jayegi.", False),
    ("english", "आशा ने अभी टिकट बुक नहीं किया है।", True),
    ("heldout", "Riya might go to Delhi on Monday.", True),
    ("heldout", "Riya will definitely go to Delhi on Monday.", False),
    ("heldout", "Riya has already bought a ticket to Delhi.", False),
    ("heldout", "Mohan said he is allergic to milk.", True),
    ("heldout", "Mohan has a life-threatening milk allergy.", False),
    ("heldout", "Riya is allergic to milk.", False),
    ("heldout", "Mohan ko milk se allergy hai, aisa usne kaha.", True),
    ("heldout", "मोहन ने दूध से एलर्जी होने की बात कही।", True),
    ("allergy", "Nisha eats peanuts every day.", False),
    ("allergy", "Nisha weighs ninety kilograms.", False),
)


def check(settings):
    client = client_for(settings)
    rows, timings = [], []
    for offset in range(0, len(CASES), MAX_CLAIMS):
        cases = CASES[offset:offset + MAX_CLAIMS]
        evidence = [{"record_id": "synthetic-" + key, "text": SOURCES[key],
                     "revision": "1", "source": "synthetic", "role": "user"}
                    for key in sorted({key for key, _, _ in cases})]
        claims = [{"text": text, "record_ids": ["synthetic-" + key],
                   "evidence": [{"record_id": "synthetic-" + key, "quote": SOURCES[key]}]}
                  for key, text, _ in cases]
        started = time.monotonic()
        try:
            output = client.verify_claims(claims, evidence=evidence)
            rejected = {item["claim_index"]: item["label"]
                        for item in output["verification"]["rejected"]}
            for index, (key, text, expected) in enumerate(cases):
                label = rejected.get(index, "supported")
                rows.append({"case": offset + index, "source": key, "claim": text,
                             "expected_supported": expected, "label": label,
                             "passed": (label == "supported") == expected})
            timings.append({"seconds": round(time.monotonic() - started, 4),
                            "input_tokens": output["input_tokens"],
                            "output_tokens": output["output_tokens"]})
        except HindsightError as error:
            rows.extend({"case": offset + index, "source": key, "claim": text,
                         "expected_supported": expected, "label": "unavailable",
                         "passed": False, "error": str(error)}
                        for index, (key, text, expected) in enumerate(cases))
    generations = []
    for key in ("allergy", "hinglish", "hindi"):
        started = time.monotonic()
        try:
            output = client.synthesize("Summarize the person's explicitly stated details, "
                "without adding advice or inferring anything.", evidence=[{
                    "record_id": "synthetic-" + key, "revision": "1", "source": "synthetic",
                    "role": "user", "text": SOURCES[key]}], max_tokens=512)
            generations.append({"source": key, "text": output["text"],
                "claims": output["claims"], "verification": output["verification"],
                "manual_review_required": True, "completed": True,
                "seconds": round(time.monotonic() - started, 4)})
        except HindsightError as error:
            generations.append({"source": key, "completed": False, "error": str(error)})
    unsupported = [row for row in rows if not row["expected_supported"]]
    supported = [row for row in rows if row["expected_supported"]]
    return {"passed": all(row["passed"] for row in rows)
                        and all(row["completed"] for row in generations),
            "cases": rows, "batches": timings, "generation": generations,
            "unsupported_false_accepts": sum(row["label"] == "supported" for row in unsupported),
            "unsupported_cases": len(unsupported),
            "supported_retained": sum(row["label"] == "supported" for row in supported),
            "supported_cases": len(supported),
            "limitation": "Synthetic smoke test only; generator and verifier share a model. "
                          "No calibrated multilingual production guarantee or Telegram test."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(load_settings(args.env_file))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("passed", "unsupported_false_accepts",
        "unsupported_cases", "supported_retained", "supported_cases", "batches", "limitation")}))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
