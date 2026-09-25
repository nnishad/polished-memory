"""C1 attachments: content-addressed bytes inside the canonical database.

The database is the only blob authority. Attachment bytes are written in the
same SQLite transaction as the row that cites them, which makes the classic
failure — a crash between writing a file and committing its record — structurally
impossible instead of something a reconciliation job hopes to notice. The cost
is that big attachments live in the database, which is the trade the initial
port deliberately makes.

Content is keyed by hash and reference counted, so a forwarded attachment is
stored once and cannot be destroyed by forgetting one of its carriers. That also
makes the count the only thing standing between an orphaned secret and the
vacuum, so every path that drops a reference drops it here.
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from ..ids import digest, now
from .evidence import EvidenceStore

__all__ = ["BlobStore", "BlobError", "Attachment", "Staged", "MAX_ATTACHMENT_BYTES",
           "MAX_ATTACHMENTS_PER_RECORD"]

# An attachment is a document, not a disk image. Past this the source is either
# misbehaving or asking us to hold something we will never render.
MAX_ATTACHMENT_BYTES = 32 * 1024 * 1024
MAX_ATTACHMENT_CHUNK = 512 * 1024
MAX_ATTACHMENTS_PER_RECORD = 32
MAX_FILENAME = 200

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_MIME = re.compile(r"^[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*$")


class BlobError(ValueError):
    pass


@dataclass(frozen=True)
class Attachment:
    id: str
    record_id: str
    position: int
    filename: str
    mime: str
    size: int
    sha256: str
    added_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Attachment":
        return cls(id=row["id"], record_id=row["record_id"], position=row["position"],
                   filename=row["filename"], mime=row["mime"], size=row["size"],
                   sha256=row["sha256"], added_at=row["added_at"])

    def as_dict(self) -> dict[str, Any]:
        """Metadata only: a listing must never pull bytes into a prompt."""
        return {"id": self.id, "record_id": self.record_id, "position": self.position,
                "filename": self.filename, "mime": self.mime, "size": self.size,
                "sha256": self.sha256, "added_at": self.added_at}


@dataclass(frozen=True)
class Staged:
    """Validated bytes with their hash and chunk plan, before any write."""

    filename: str
    mime: str
    sha256: str
    size: int
    chunks: tuple[bytes, ...]
    # Where the source says this one sits in the order. None means "in the order
    # given", which is not the same claim as "position 3 of four".
    position: int | None = None


def normalize_filename(value: Any) -> str:
    """Make a source-supplied name inert.

    It is never used as a path, but it is shown to a model and to a human, so a
    name carrying control characters or a traversal attempt is cleaned rather
    than trusted to stay where it is.
    """
    text = "" if value is None else str(value)
    text = text.replace("\\", "/").rsplit("/", 1)[-1]
    text = _CONTROL.sub("", text).strip()
    if not text:
        return "unnamed"
    return text[:MAX_FILENAME]


def normalize_mime(value: Any) -> str:
    declared = ("" if value is None else str(value)).split(";", 1)[0].strip().lower()
    if not _MIME.match(declared):
        raise BlobError(
            f"mime {value!r} is not a 'type/subtype' token; an unparseable type is refused "
            "at ingress rather than recorded as a guess")
    return declared


def stage_bytes(data: Any, *, filename: Any, mime: Any,
                position: int | None = None) -> Staged:
    """Validate and hash. Touches no database, so a bad payload never opens a lock."""
    if isinstance(data, str):
        raise BlobError("attachment bytes must be bytes, not text")
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise BlobError(f"attachment bytes must be bytes, got {type(data).__name__}")
    raw = bytes(data)
    if not raw:
        raise BlobError("an empty attachment is a source bug, not a zero-length document")
    if len(raw) > MAX_ATTACHMENT_BYTES:
        raise BlobError(
            f"attachment is {len(raw)} bytes, above the {MAX_ATTACHMENT_BYTES} byte ceiling")
    declared = normalize_mime(mime)
    chunks = tuple(raw[offset:offset + MAX_ATTACHMENT_CHUNK]
                   for offset in range(0, len(raw), MAX_ATTACHMENT_CHUNK))
    if position is not None and (not isinstance(position, int) or isinstance(position, bool)
                                 or not 0 <= position < MAX_ATTACHMENTS_PER_RECORD):
        raise BlobError(
            f"attachment position must be an integer below {MAX_ATTACHMENTS_PER_RECORD}")
    return Staged(filename=normalize_filename(filename), mime=declared,
                  sha256=hashlib.sha256(raw).hexdigest(), size=len(raw), chunks=chunks,
                  position=position)


class BlobStore:
    def __init__(self, store: EvidenceStore):
        self.store = store
        self.db = store.db

    # -- write ---------------------------------------------------------------

    def attach(self, record_pk: str, payloads: Sequence[dict[str, Any]], *,
               db: sqlite3.Connection | None = None) -> list[Attachment]:
        """Attach ordered payloads to one record. Requires an ambient transaction.

        Idempotent per (record, position, content): replaying the same page of a
        source must not double-count a reference or fork a second attachment row.
        """
        if not isinstance(record_pk, str) or not record_pk.startswith("rec_"):
            raise BlobError(f"attachment target must be a canonical record id, got {record_pk!r}")
        if len(payloads) > MAX_ATTACHMENTS_PER_RECORD:
            raise BlobError(
                f"{len(payloads)} attachments exceed the {MAX_ATTACHMENTS_PER_RECORD} per "
                "record ceiling")
        connection = db or self.db
        _require_transaction(connection)
        # Read on the caller's connection: in a combined commit the record was
        # created inside this same transaction and is invisible from anywhere else.
        if not connection.execute("SELECT 1 FROM records WHERE id=?",
                                  (record_pk,)).fetchone():
            raise BlobError(f"cannot attach to unknown record {record_pk!r}")
        staged = [
            stage_bytes(item.get("data"), filename=item.get("filename"),
                        mime=item.get("mime"), position=item.get("position"))
            for item in _as_mapping_list(payloads)
        ]
        return self._write(connection, record_pk, staged)

    def _write(self, connection: sqlite3.Connection, record_pk: str,
               staged: Sequence[Staged]) -> list[Attachment]:
        written: list[Attachment] = []
        for index, item in enumerate(staged):
            position = index if item.position is None else item.position
            if any(existing.position == position for existing in written):
                raise BlobError(f"position {position} is claimed twice in one write")
            existing = connection.execute(
                "SELECT * FROM attachments WHERE record_id=? AND position=?",
                (record_pk, position)).fetchone()
            if existing:
                if existing["sha256"] != item.sha256:
                    # The bytes behind a cited position can change; the citation
                    # cannot be allowed to point at something else.
                    raise BlobError(
                        f"position {position} of {record_pk} already holds different bytes; "
                        "ingest a new revision instead of rewriting an attachment")
                # Same bytes, same place: report what is stored, including the
                # name and timestamp recorded by whoever put it there.
                written.append(Attachment.from_row(existing))
                continue
            attachment = _attachment_id(record_pk, position, item.sha256)
            connection.execute(
                "INSERT INTO blob_contents(sha256, size, refs, first_seen) VALUES(?,?,0,?) "
                "ON CONFLICT(sha256) DO NOTHING",
                (item.sha256, item.size, now()),
            )
            for index, chunk in enumerate(item.chunks):
                connection.execute(
                    "INSERT OR IGNORE INTO blob_chunks(sha256, chunk_index, data) VALUES(?,?,?)",
                    (item.sha256, index, chunk),
                )
            added_at = now()
            connection.execute(
                "INSERT INTO attachments(id, record_id, position, filename, mime, size, sha256, "
                "added_at) VALUES(?,?,?,?,?,?,?,?)",
                (attachment, record_pk, position, item.filename, item.mime, item.size,
                 item.sha256, added_at),
            )
            # The reference count moves with the row that creates it, so a
            # rollback cannot leave bytes that look referenced, or a committed
            # row whose bytes were already collected.
            connection.execute(
                "UPDATE blob_contents SET refs=refs+1 WHERE sha256=?", (item.sha256,))
            written.append(Attachment(id=attachment, record_id=record_pk, position=position,
                                      filename=item.filename, mime=item.mime, size=item.size,
                                      sha256=item.sha256, added_at=added_at))
        return written

    # -- read ----------------------------------------------------------------

    def get(self, attachment_id: str) -> Attachment | None:
        row = self.db.execute(
            "SELECT * FROM attachments WHERE id=?", (attachment_id,)).fetchone()
        return Attachment.from_row(row) if row else None

    def list_for(self, record_pk: str) -> list[Attachment]:
        """Attachments of a record the caller may still see, in source order."""
        if not self.store.live_and_visible(record_pk):
            return []
        rows = self.db.execute(
            "SELECT * FROM attachments WHERE record_id=? ORDER BY position",
            (record_pk,)).fetchall()
        return [Attachment.from_row(row) for row in rows]

    def read(self, attachment_id: str) -> bytes:
        """Return verified bytes, or raise. An invisible record has no bytes.

        The hash is recomputed on every read: a store that silently returns
        damaged bytes is worse than one that fails, because the damage becomes
        a fact the owner has to disprove later.
        """
        row = self.db.execute(
            "SELECT * FROM attachments WHERE id=?", (attachment_id,)).fetchone()
        if row is None:
            raise BlobError(f"unknown attachment {attachment_id!r}")
        if not self.store.live_and_visible(row["record_id"]):
            raise BlobError(f"attachment {attachment_id!r} is not retrievable")
        chunks = self.db.execute(
            "SELECT chunk_index, data FROM blob_chunks WHERE sha256=? ORDER BY chunk_index",
            (row["sha256"],)).fetchall()
        raw = b"".join(bytes(chunk["data"]) for chunk in chunks)
        if len(chunks) == 0 or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise BlobError(
                f"attachment {attachment_id!r} does not match its recorded hash; "
                "the stored bytes are damaged and were not handed out")
        if len(raw) != row["size"]:
            raise BlobError(f"attachment {attachment_id!r} has the wrong length")
        return raw

    def summarize(self, record_pk: str) -> dict[str, int]:
        """Byte totals for a blast radius. Counts what exists, visible or not."""
        row = self.db.execute(
            "SELECT count(*) AS files, COALESCE(sum(size), 0) AS bytes FROM attachments "
            "WHERE record_id=?", (record_pk,)).fetchone()
        return {"files": int(row["files"] or 0), "bytes": int(row["bytes"] or 0)}

    # -- destruction ---------------------------------------------------------

    def release(self, record_pks: Iterable[str], *,
                db: sqlite3.Connection | None = None) -> dict[str, int]:
        """Drop every reference held by these records and collect what is left.

        Requires an ambient transaction and shares it with the tombstone, so a
        forgotten record cannot leave its attachment readable in one and
        unreferenced in the other.
        """
        targets = [pk for pk in record_pks if pk]
        if not targets:
            return {"attachments": 0, "contents": 0, "bytes": 0}
        connection = db or self.db
        _require_transaction(connection)
        placeholders = ",".join("?" * len(targets))
        rows = connection.execute(
            f"SELECT id, sha256, size FROM attachments WHERE record_id IN ({placeholders})",
            targets).fetchall()
        if not rows:
            return {"attachments": 0, "contents": 0, "bytes": 0}
        released: dict[str, int] = {}
        for row in rows:
            connection.execute("DELETE FROM attachments WHERE id=?", (row["id"],))
            released[row["sha256"]] = released.get(row["sha256"], 0) + 1
        for sha256, count in released.items():
            connection.execute(
                "UPDATE blob_contents SET refs=refs-? WHERE sha256=?", (count, sha256))
        collected = self._collect(connection)
        return {"attachments": len(rows), "contents": collected,
                "bytes": sum(row["size"] for row in rows)}

    def _collect(self, connection: sqlite3.Connection) -> int:
        """Delete unreferenced content. A record still holding a reference is never touched."""
        rows = connection.execute(
            """
            SELECT c.sha256, c.refs FROM blob_contents c
            WHERE c.refs = 0
              AND NOT EXISTS (SELECT 1 FROM attachments a WHERE a.sha256 = c.sha256)
            """).fetchall()
        for row in rows:
            connection.execute("DELETE FROM blob_chunks WHERE sha256=?", (row["sha256"],))
            connection.execute("DELETE FROM blob_contents WHERE sha256=?", (row["sha256"],))
        return len(rows)

    def reconcile(self) -> dict[str, int]:
        """Recount references from the attachment rows and collect what nobody holds.

        This exists for the case the transaction cannot cover: a database copied
        or restored by hand. It is a repair, not the normal path, and it never
        deletes a byte that an attachment still names.
        """
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute("""
                UPDATE blob_contents SET refs = (
                    SELECT count(*) FROM attachments a WHERE a.sha256 = blob_contents.sha256)
                """)
            collected = self._collect(self.db)
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        left = self.db.execute("SELECT count(*) FROM blob_contents").fetchone()[0]
        chunks = self.db.execute("SELECT count(*) FROM blob_chunks").fetchone()[0]
        return {"contents_collected": collected, "contents": int(left),
                "chunks": int(chunks)}


def _attachment_id(record_pk: str, position: int, sha256: str) -> str:
    return "blb_" + digest([record_pk, position, sha256])[:32]


def _require_transaction(connection: sqlite3.Connection) -> None:
    """Refuse to run outside a transaction the caller owns.

    Half a release is the failure this guards: bytes collected while the row
    that referenced them is still in an open transaction, or a reference count
    that moved on a rollback.
    """
    if not connection.in_transaction:
        raise BlobError("blob writes require an ambient transaction")


def _as_mapping_list(payloads: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    items = list(payloads)
    for item in items:
        if not isinstance(item, dict):
            raise BlobError("each attachment is an object with data, filename and mime")
    return items
