"""Read-only, caller-scoped verification of memory claims for Hermes tools."""
from __future__ import annotations

import json

from ..backend.hindsight_client import HindsightError
from ..storage.identity import IdentityStore, evidence_accounts, in_scope
from .faithfulness import (MAX_CLAIMS, VERIFICATION_TOKENS, validate_publication,
                           canonical_evidence, normalize_claims)
from .budgets import Budget, BudgetExhausted, Budgets
from .instance_gate import instance_gate
from .routes import build_routes
from .synthesis import ScopedSynthesizer


def verify_memory_claims(settings, store, claims, *, account_id=None, client=None):
    """Resolve sources ourselves; never accept caller-supplied evidence/authority.

    A model verdict is scoped to this snapshot. No canonical or derived records
    are written; visibility/revision changes during inference withhold output.
    """
    if not settings.inference_enabled or settings.capture_only:
        raise HindsightError("memory claim verification requires enabled inference")
    if not isinstance(claims, list) or not 1 <= len(claims) <= MAX_CLAIMS:
        raise HindsightError("bounded memory claims required")
    refs = set()
    for claim in claims:
        if (not isinstance(claim, dict) or not isinstance(claim.get("record_ids"), list)
                or not 1 <= len(claim["record_ids"]) <= 32
                or any(not isinstance(ref, str) or not 1 <= len(ref) <= 200
                       for ref in claim["record_ids"])):
            raise HindsightError("bounded canonical record references required")
        refs.update(claim["record_ids"])
    identity = IdentityStore(store)
    identifiers = set(identity.group(account_id)) if account_id else set()
    stamp = store.watermark()
    evidence = []
    for ref in sorted(refs):
        record = store.get(ref)
        if (record is None or not store.live_and_visible(ref)
                or not in_scope(evidence_accounts(identity, record), identifiers)):
            # Do not expose whether another person's source exists.
            raise HindsightError("claim evidence is unavailable in the caller's scope")
        evidence.append({"record_id": record.id, "revision": record.revision,
                         "text": record.text, "source": record.source,
                         "role": record.metadata.get("role"), "occurred_at": record.occurred_at})
    route = build_routes(settings, credentials=settings.route_credentials).by_name("foreground")
    normalize_claims(claims, canonical_evidence(evidence))
    # Do not infer permission from availability of a model endpoint. Like
    # formation/summarization, respect the shared daily budget and operator hold
    # before any dispatch. The HTTP gate accounts the actual model hop.
    estimate = len(json.dumps({"claims": claims, "evidence": evidence}, ensure_ascii=False).encode())
    with instance_gate(settings) as gate:
        if gate.paused:
            raise HindsightError("all inference is paused by the operator")
        try:
            Budgets(gate.store, daily={route.resource: Budget(settings.background_budget_tokens)}).admit(
                route.resource, estimated_tokens=estimate + 4096 + VERIFICATION_TOKENS)
        except BudgetExhausted as error:
            raise HindsightError(str(error)) from error
    holder = client or ScopedSynthesizer(base_url=settings.admission_url,
        credential=route.credential, model=settings.text_route.model,
        # Explicit tools do not run under Hermes's eight-second prefetch hook.
        # A transport timeout still withholds output; no model retry/fallback.
        timeout=180.0)
    outcome = holder.verify_claims(claims, evidence=evidence,
                                  max_tokens=min(VERIFICATION_TOKENS, route.max_output_tokens))
    if store.watermark() != stamp:
        raise HindsightError("canonical authority changed during verification; output withheld")
    validate_publication(outcome, evidence, allow_empty=True)
    return {**outcome, "ok": True, "all_supported": len(outcome["claims"]) == len(claims),
            "source_watermark": list(stamp) if isinstance(stamp, tuple) else stamp,
            "instruction": "Use only checked text for memory assertions. Do not add factual clauses. "
                           "Omit rejected claims or clarify important unknowns through Hermes. "
                           "These are model support decisions, not proof of source truth."}
