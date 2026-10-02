"""Bounded, local-only resolution of backend facts and observation support.

No backend claim grants access. The caller authorizes the resulting canonical
records separately; every declared support reference must resolve before that.
"""
from __future__ import annotations

import json
from collections.abc import Mapping


class SupportError(ValueError):
    pass


class SupportResolver:
    MAX_REFERENCES = 500
    MAX_DEPTH = 8
    MAX_BYTES = 262_144

    def __init__(self, store, *, bank_id: str | None = None, source_facts=()):
        self.store = store
        self.bank_id = bank_id
        if bank_id and store.db.execute("SELECT 1 FROM sqlite_master WHERE name='projection_generations'").fetchone():
            generation = store.db.execute("SELECT state,epoch FROM projection_generations "
                                          "WHERE backend='hindsight' AND bank_id=?", (bank_id,)).fetchone()
            if generation and (generation["state"] != "active" or generation["epoch"] != store.epoch()):
                raise SupportError("support bank has no active current projection generation")
        if not isinstance(source_facts, (tuple, list)) or len(source_facts) > self.MAX_REFERENCES:
            raise SupportError("source-fact manifest exceeds its bound")
        if len(json.dumps(source_facts, ensure_ascii=False).encode()) > self.MAX_BYTES:
            raise SupportError("source-fact manifest exceeds its byte bound")
        self.facts = {}
        for fact in source_facts:
            if not isinstance(fact, Mapping):
                raise SupportError("source-fact manifest contains a non-object")
            identifier = self._id(fact.get("id"))
            if identifier in self.facts:
                raise SupportError("source-fact manifest contains duplicate identities")
            self.facts[identifier] = fact

    @staticmethod
    def _id(value):
        if not isinstance(value, str) or not value.strip() or len(value) > 200:
            raise SupportError("support reference is not a bounded identity")
        return value

    def resolve(self, fact: Mapping) -> tuple[str, ...]:
        visited = set()
        memo = {}
        references = 0

        def count():
            nonlocal references
            references += 1
            if references > self.MAX_REFERENCES:
                raise SupportError("support traversal exceeds its reference bound")

        def document(identifier):
            count()
            identifier = self._id(identifier)
            sql = ("SELECT record_id, revision FROM backend_documents WHERE document_id=? "
                   "AND backend='hindsight' AND state='verified' AND desired_epoch=?")
            params = [identifier, self.store.epoch()]
            if self.bank_id:
                sql += " AND bank_id=?"
                params.append(self.bank_id)
            rows = self.store.db.execute(sql, params).fetchall()
            if len(rows) != 1:
                raise SupportError("support document has no unique verified bank/epoch mapping")
            record = self.store.get(rows[0]["record_id"])
            if record is None or record.revision != rows[0]["revision"]:
                raise SupportError("support document names a stale revision")
            return record.id

        def walk(node, depth=0):
            if depth > self.MAX_DEPTH or not isinstance(node, Mapping):
                raise SupportError("support traversal is too deep or malformed")
            if node.get("bank_id") and self.bank_id and node["bank_id"] != self.bank_id:
                raise SupportError("support belongs to another bank")
            records = set()
            for key in ("record_id", "document_id"):
                if node.get(key) is not None:
                    if key == "document_id":
                        records.add(document(node[key]))
                    else:
                        count()
                        records.add(self._id(node[key]))
            for key in ("record_ids", "document_ids", "source_fact_ids"):
                values = node.get(key)
                if values is None:
                    values = ()
                if not isinstance(values, (list, tuple)) or len(values) > self.MAX_REFERENCES:
                    raise SupportError("support references are not a bounded sequence")
                for value in values:
                    value = self._id(value)
                    if key == "document_ids":
                        records.add(document(value))
                    elif key == "record_ids":
                        count()
                        records.add(value)
                    else:
                        count()
                        if value in visited:
                            raise SupportError("cyclic observation support")
                        if value not in self.facts:
                            raise SupportError("observation support is missing")
                        if value not in memo:
                            visited.add(value)
                            memo[value] = walk(self.facts[value], depth + 1)
                            visited.remove(value)
                        records.update(memo[value])
            if not records:
                raise SupportError("backend claim has no canonical support")
            return records

        records = walk(fact)
        for identifier in records:
            if not self.store.live_and_visible(identifier) or self.store.get(identifier) is None:
                raise SupportError("support is hidden, deleted, or unknown")
        return tuple(sorted(records))
