# Source adapters

An adapter is the only place where a foreign format meets this archive, and the contract
is deliberately small: **declare what the source can do, prove you can reach it, hand over
bounded pages of versioned envelopes, and never open a store.**

Everything below is what `src/hermes_memory/sources/` actually does. Where the plan asks
for a step that this build has not earned yet, it is named as absent rather than described
aspirationally — an adapter document that overstates what a connector does becomes the
reason somebody trusts a record that was never really read.

## The contract

```python
class SourceAdapter:
    source: str                      # stamps every envelope, and cannot be left blank
    capabilities: Capabilities       # history, live, deletion_events, revision_history,
                                     # max_records_per_page, max_bytes_per_page

    def check(self) -> dict          # cheap reachability and permission; reads no content
    def read_page(self, cursor) -> Page   # one bounded page: envelopes + skipped + next cursor
    def envelope(self, **values) -> dict  # stamps this adapter's source name
    def read_all(self, *, max_pages=10_000)  # drains to the end, and stops if there is no end
```

`Capabilities` is a declaration, not an optimisation. Discovery drives the plan: a source
that cannot report deletions can never promise that a forgetting reached it, and that has
to be visible before the first read rather than discovered during an audit. `check()` is
what `hermes-memory sources` runs; `read_page()` is what the sync controller drives under a
fence.

**There is no `normalize(raw)` callback.** Normalization is shared code an adapter calls
from wherever its own format actually arrives — `normalize_text` (decodes strictly, returns
`None` rather than a mojibake guess, and redacts secret-shaped substrings),
`normalize_time` (an instant, its precision, and a note when the source said something
imprecise), `redact_secrets`, `revocation`. A hook the framework could claim to have run
while every adapter shaped its records inside `read_page` anyway would be one more thing to
audit and one less thing to believe.

## The envelope

A page carries versioned envelopes, and an envelope is the canonical record shape the store
accepts: `source`, `source_id`, `revision`, `kind`, `text`, `observed_at`, `occurred_at`,
`occurred_precision`, `metadata`. Two rules do most of the safety work:

- **(source, source_id, revision) is immutable.** A correction is a new revision, never an
  update, so "what did we believe on the 14th" stays answerable and an overwrite cannot
  quietly rewrite history. Two different bodies under one revision are refused.
- **A page reports what it skipped.** `Skipped` is a countable, reason-carrying gap: a
  message that could not be decoded, a part over the byte ceiling, a page whose cursor
  expired. `CursorExpired` is raised, not swallowed, so a connector resumes from a fence it
  still holds rather than from wherever the remote side decided to start again. A gap is
  reported under the id of the *thing* that was missing, and an adapter that names a part of a
  thing `whole#part` has its debt closed by any part arriving: one event reported as a whole and
  delivered later as its two sides is the source handing the thing over, and matching only the
  bare id would keep a gap open forever against an id no envelope can ever carry.

Checkpoints are opaque strings owned by the source. The store keeps them, compares them and
refuses to move backwards through them; it never parses one.

## What is implemented

| Adapter | Reads | Declares | Known limits |
| --- | --- | --- | --- |
| `email.py` | IMAP-ish mailboxes and `.eml`/`mbox` trees, thread participants and headers | history, revision per `Message-ID` + body digest | attachment **bytes are named, not stored**: sizes, names and MIME are recorded, the content is not ingested |
| `whatsapp_export.py` | a exported-chat directory (`_metadata.json` plus per-chat `.txt`/`.json`) | history only; an export is a snapshot, not a live box | no deletion events: forgetting a message there is proven locally, never at the source |
| `mcp.py` | a tool that returns pages of records, mapped by a declared field map | whatever the server's declared capabilities say | an unmapped `record_id` or `revision` is a refusal, not a guess |
| `structured.py` | CSV/TSV/JSONL with a declared column map | history; revision from a declared column or a content digest | a row with no time is stored as undated, and an undated record cannot satisfy a time window |
| `files.py` | a tree of markdown/text notes with front matter | history; revision from mtime+size or content digest | renames are new records; the adapter does not infer that two paths are one note |
| `sdk.py` | the host's own capture stream, handed over as a `drain(after, limit)` callable | live and history; no deletion events, no revision history | reads four event kinds — a conversation turn, a `pre_compress` checkpoint, a `session_end` transcript and a mirrored native note; any other kind is reported as a gap under the event's own id, so learning it later delivers the event instead of losing it |
| `capture_spool.py` | not an adapter: the consumer side of the plugin's durable spool, which `sdk.py` reads through a drain | — | position is the spool's own rowid, never its ids; a row is retired as `settled` — the plugin's word, the one its compaction reclaims — only once a later read has passed it |
| `runtime.py` | not an adapter: drives one adapter under one lease, and stops for a reason | — | — |
| `sync.py` | not an adapter: the fence, the checkpoint ledger, replay and gap accounting | — | — |

## Not yet, stated plainly

- **No adapter ingests attachment bytes.** The store's side of that is open and tested: an
  envelope that brings `data` has it validated outside any lock, content-hashed, chunked and
  written in the *same* transaction as the record it belongs to, with the ingestion receipt
  keeping the name, type, size and hash and never the bytes, and a replay of the same
  revision restoring a blob that was lost in between. What no adapter does is *read*
  somebody's file in order to supply those bytes: staging arbitrary attachments out of a
  mailbox is a larger authorization question than a connector should answer by itself. Until
  an authorized reader exists, `hermes-memory summarize` and the context packet can say what
  an attachment *was*, never what it said.
- **No adapter is enabled by discovery.** `hermes-memory sources --connect` and the import
  door record a source's declared policy (`local-only`, `private-api`, `disabled`) before
  anything is read, and nothing reads a source the owner has not named.
- **Deletion events are not faked.** Where a source cannot report them, an erasure
  obligation for that source is verified against the local tombstone and the derived
  backend, and the source-side limit stays visible in the erasure ledger.

## Writing a new one

1. Subclass `SourceAdapter`, set `source`, and declare `capabilities` truthfully — a
   capability you cannot prove is a capability you should not list.
2. `check()` must not read content: a permission probe that pulls the archive is a leak
   hiding as a health check.
3. `read_page()` gets a cursor it did not invent, returns at most the declared page bounds,
   and reports every dropped item as `Skipped`.
4. Build envelopes through `self.envelope(...)`; never commit anything. The sync controller
   owns the store, the fence and the epoch.
5. Add the adapter to `deployment/compatibility.json` if it changes what a version of this
   build can claim, and write the fault cases (expired cursor, duplicate revision with
   different bytes, undated item, oversized page) before the happy path.
