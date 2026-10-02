"""Versioned, shadow-bank projection rebuilds with reviewed atomic cutover.

This ledger does not provision banks or invoke models. Formation owns submission,
the document map owns verification, and every historical bank stays represented
for erasure. No generation mutates a canonical revision or reuses another bank.
"""
from __future__ import annotations

import json
import re

from ..ids import digest, now
from ..storage.evidence import EvidenceError


class GenerationRegistry:
    def __init__(self, store, *, backend="hindsight", family_bank="hermes", owner_principal=None):
        self.store, self.db = store, store.db
        self.backend, self.family_bank = backend, family_bank
        self.owner_principal = owner_principal

    def _owner(self, actor):
        if not self.owner_principal or actor != self.owner_principal:
            raise EvidenceError("projection generation changes require the configured owner")

    def active(self):
        row = self.db.execute(
            "SELECT * FROM projection_generations WHERE backend=? AND family_bank=? "
            "AND state='active' AND epoch=?", (self.backend, self.family_bank,
                                               self.store.epoch())).fetchone()
        return dict(row) if row else None

    def get(self, identifier):
        row = self.db.execute("SELECT * FROM projection_generations WHERE id=? AND backend=? "
                              "AND family_bank=?", (identifier, self.backend,
                                                   self.family_bank)).fetchone()
        return dict(row) if row else None

    def plan(self, manifest):
        if not isinstance(manifest, dict) or manifest.get("version") != "processor-manifest-v1":
            raise EvidenceError("a rebuild requires a versioned processor manifest")
        # Exact declared inputs, including explicitly unknown fields. Do not claim
        # that a declared vector or prompt contract has been probed at runtime.
        encoded = json.dumps(manifest, sort_keys=True, ensure_ascii=False)
        if len(encoded.encode()) > 16384:
            raise EvidenceError("processor manifest exceeds its byte bound")
        if not re.fullmatch(r"[A-Za-z0-9-]{1,100}", self.family_bank):
            raise EvidenceError("generation family bank must be a bounded opaque bank name")
        epoch = self.store.epoch()
        identifier = "gen-" + digest([self.backend, self.family_bank, epoch, manifest])[:32]
        bank = self.family_bank + "-g-" + identifier[4:20]
        active = self.active()
        result = {"version": "generation-plan-v1", "generation_id": identifier,
                  "family_bank": self.family_bank, "bank_id": bank, "backend": self.backend,
                  "epoch": epoch, "manifest": manifest, "watermark": list(self.store.watermark()),
                  "active_generation": active["id"] if active else None,
                  "existing": self.get(identifier) is not None,
                  "records": self._live_count()}
        result["review_digest"] = digest(result)
        return result

    def prepare(self, manifest, *, actor, review):
        self._owner(actor)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            plan = self.plan(manifest)
            if review != plan["review_digest"]:
                raise EvidenceError("rebuild review changed; read the current generation plan")
            if not self.db.execute("SELECT 1 FROM projection_generations WHERE backend=? "
                                   "AND bank_id=?", (self.backend, self.family_bank)).fetchone():
                legacy = self.db.execute("SELECT max(desired_epoch) FROM backend_documents "
                                         "WHERE backend=? AND bank_id=?",
                                         (self.backend, self.family_bank)).fetchone()[0]
                if legacy is not None:
                    legacy_id = "legacy:" + self.backend + ":" + self.family_bank
                    self.db.execute("INSERT INTO projection_generations"
                        "(id,backend,family_bank,bank_id,epoch,manifest,state,legacy,created_at) "
                        "VALUES(?,?,?,?,?,'{\"version\":\"legacy-unknown\"}',?,1,?)",
                        (legacy_id, self.backend, self.family_bank, self.family_bank, legacy,
                         "active" if legacy == self.store.epoch() else "retired", now()))
                    self.db.execute("UPDATE backend_documents SET generation_id=? "
                        "WHERE backend=? AND bank_id=? AND generation_id IS NULL",
                        (legacy_id, self.backend, self.family_bank))
            self.db.execute(
                "INSERT OR IGNORE INTO projection_generations"
                "(id,backend,family_bank,bank_id,epoch,manifest,state,created_at) "
                "VALUES(?,?,?,?,?,?,'building',?)",
                (plan["generation_id"], self.backend, self.family_bank, plan["bank_id"],
                 plan["epoch"], json.dumps(manifest, sort_keys=True), now()))
            self.store._audit("projection_generation_prepare", plan["generation_id"],
                              {"actor": actor, "bank_id": plan["bank_id"]})
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return self.get(plan["generation_id"])

    def _live_count(self):
        return int(self.db.execute("SELECT count(*) FROM records r WHERE r.deleted=0 "
            "AND NOT EXISTS(SELECT 1 FROM record_visibility v WHERE v.record_id=r.id "
            "AND v.hidden=1)").fetchone()[0])

    def cutover_plan(self, identifier):
        generation = self.get(identifier)
        if generation is None:
            raise EvidenceError("unknown projection generation")
        blocking = []
        if generation["legacy"]:
            blocking.append("unknown legacy contracts cannot be cut over as a verified rebuild")
        if generation["epoch"] != self.store.epoch():
            blocking.append("generation belongs to a stale memory epoch")
        manifest = json.loads(generation["manifest"])
        if (type(manifest.get("embedding_dimensions")) is not int
                or not 1 <= manifest["embedding_dimensions"] <= 65536
                or not manifest.get("bank_prompt_config")):
            blocking.append("embedding dimensions and bank prompt contract remain unverified")
        rows = self.db.execute(
            "SELECT b.*, r.text FROM backend_documents b JOIN records r ON r.id=b.record_id "
            "WHERE b.generation_id=? AND b.backend=? AND b.bank_id=? AND b.state='verified' "
            "AND b.desired_epoch=? AND b.revision=r.revision AND r.deleted=0 "
            "AND NOT EXISTS(SELECT 1 FROM record_visibility v WHERE v.record_id=r.id AND v.hidden=1)",
            (identifier, self.backend, generation["bank_id"], self.store.epoch())).fetchall()
        valid = 0
        for row in rows:
            expected = input_manifest(self.store, row["record_id"], row["revision"])
            try:
                matches = json.loads(row["input_manifest"] or "null") == expected
            except (ValueError, TypeError):
                matches = False
            valid += bool(matches and row["retain_payload_digest"])
        total = self._live_count()
        if valid != total:
            blocking.append(f"current verified input coverage is incomplete: {valid}/{total}")
        # Unfinished deletion is an obligation, not an inferred success from an
        # absent local row. Cutover must not hide it in an old bank.
        if self.db.execute("SELECT 1 FROM erasure_targets WHERE state!='verified' LIMIT 1").fetchone():
            blocking.append("unverified erasure obligations remain")
        if self.db.execute("SELECT 1 FROM processing_jobs WHERE generation_id=? AND state NOT IN "
                           "('succeeded','cancelled','quarantined') LIMIT 1", (identifier,)).fetchone():
            blocking.append("generation still has unfinished processing jobs")
        result = {"version": "generation-cutover-v1", "generation_id": identifier,
                  "bank_id": generation["bank_id"], "epoch": self.store.epoch(),
                  "watermark": list(self.store.watermark()), "verified": valid,
                  "records": total, "blocking": blocking}
        result["review_digest"] = digest(result)
        return result

    def activate(self, identifier, *, actor, review):
        self._owner(actor)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            plan = self.cutover_plan(identifier)
            if review != plan["review_digest"] or plan["blocking"]:
                raise EvidenceError("cutover refused: changed review or incomplete lifecycle/coverage gates")
            self.db.execute("UPDATE projection_generations SET state='retired' WHERE backend=? "
                            "AND family_bank=? AND state='active' AND id!=?",
                            (self.backend, self.family_bank, identifier))
            self.db.execute("UPDATE projection_generations SET state='active',activated_at=? WHERE id=?",
                            (now(), identifier))
            self.store._audit("projection_generation_activate", identifier, {"actor": actor})
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return self.get(identifier)


def input_manifest(store, record_id, revision):
    """Single-record, full-span input. Episodes must not fake a shared revision."""
    record = store.get(record_id)
    if record is None or record.revision != revision or not store.live_and_visible(record_id):
        raise EvidenceError("projection input is not a current visible canonical revision")
    return {"version": "projection-input-v1", "epoch": store.epoch(),
            "inputs": [{"record_id": record_id, "revision": revision,
                        "span": [0, len(record.text)]}]}


def recall_bank(store, configured_bank, *, backend="hindsight"):
    """Resolve the active current binding; never fall back after a known reset."""
    rows = store.db.execute("SELECT bank_id,state,epoch FROM projection_generations "
                            "WHERE backend=? AND family_bank=?", (backend, configured_bank)).fetchall()
    if not rows:
        return configured_bank  # undeclared legacy profile, not a guessed generation
    for row in rows:
        if row["state"] == "active" and row["epoch"] == store.epoch():
            return row["bank_id"]
    return None
