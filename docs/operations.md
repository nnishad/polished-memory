# Operating hermes-memory

Two questions, kept apart on purpose:

- **`status`** answers *what is happening*. It is watched, it is cheap, and it never acts.
- **`doctor`** answers *why is it not*. It is run when something is wrong, and a run of it
  must not make the situation worse.

Both read the canonical store through a handle that cannot change it, so running them
against a live installation is not an action. Both take `--hermes-home <profile home>` to
answer for the memory enrolled for that Hermes profile instead of the default one, and both
print JSON on stdout and a reason on stderr when they refuse.

## What is happening

```sh
hermes-memory status
hermes-memory status --hermes-home ~/.hermes/profiles/work
```

Nine stages report separately, and each one answers only for itself: `capture`,
`raw_indexing`, `observations`, `summaries`, `goals`, `analysis`, `delivery`, `backend`,
`resource_gate`. Their states are `configured`, `operational`, `degraded`, `paused`,
`unconfigured` and `disabled` — and `configured` is never written as `operational`, because
a route that exists is not a route that answered.

The report also carries three things an operator reads next:

- `erasure_backlog` — obligations that are not verified gone everywhere, tombstones, and how
  many intents are `awaiting_owner`.
- `awaiting_owner` — the identity candidates, candidate assertions and erasure intents that
  only a human may decide. When any exist, a note names the command that can decide them.
- `queue` — the age of the oldest waiting job against the allowance, with
  `"unattended": false` beside it. A queue that is not being drained by anything is said as
  that: *only `hermes-memory form` works this queue — nothing drains it by itself*.

Nothing here is a health check that calls a model. `backend` reads the projection ledger,
which is true without a socket; asking whether the backend is answering is the probe below.

## Why it is not

```sh
hermes-memory doctor
hermes-memory doctor --probe             # one request: is anything answering
hermes-memory doctor --synthetic-probe   # one bounded synthetic retain and recall
```

Thirteen checks run by default: layout, database, schema, configuration, coverage, queue,
provenance, erasure, delivery, gate, a credential scan over the newest records, leases, and
the projection ledger. `--probe` adds a connectivity request and `--synthetic-probe` a
bounded round trip in a bank of its own; the report says which probes ran, so a later reader
can tell what was actually looked at. Exit is `0`, or `1` when a check reached `fail`;
refusals (an unreadable configuration, an unenrolled profile) exit `2` with the reason on
stderr.

Every finding carries a remedy where one exists. "degraded" is not an answer; "run
`hermes-memory form` to work this queue, or resume a stage an operator paused" is.

## The decisions that belong to the owner

An agent may propose a candidate, open a forgetting preview, and report what it found. It
may not confirm any of it, and the library refuses a caller who is not the named owner
principal. `owner` is the other side of that arrangement:

```sh
hermes-memory owner --list
hermes-memory owner --confirm-forgetting erase_1f3c… --digest <the 64 characters listed>
hermes-memory owner --confirm-identity cand_9b2… --reason "the owner recognised both"
hermes-memory owner --reject-identity cand_9b2… --reason "two people"
hermes-memory owner --revoke-edge iedge_77a… --reason "the second address is a colleague's"
hermes-memory owner --confirm-assertion asr_41c… --reason "yes, that is my habit"
hermes-memory owner --retract-assertion asr_41c… --reason "one late invoice undoes it"
```

`--list` shows what awaits, per enrolled memory, **counted rather than quoted**: intent and
candidate ids, who proposed them, the digest that signs a preview, and the size of the blast
radius — not the record identifiers or texts behind them, which are read with `explain` by
somebody allowed to see them. The listing opens every store read-only; a command run in order
to look must not be the one that migrates it.

Forgetting is confirmed against the digest of the preview that was shown, because a digest
proves which selection was seen and not that `approved=true` was set. Identity and assertion
decisions require a reason, and it is written into the ledger beside the decision: they
outlive the conversation that made them. With more than one memory enrolled, a decision has
to name its profile — which store an identity belongs to is never guessed.

`HERMES_MEMORY_OWNER_PRINCIPAL` is the name these checks compare against, and it is unset by
default. With no owner named, `--list` still works and every decision is refused, which is
the correct shape for a machine nobody has told who it belongs to.

## Holding a stage

```sh
hermes-memory pause --scope inference --actor "$USER" --reason "away for the weekend"
hermes-memory pause --scope delivery --actor "$USER"
hermes-memory pause --scope inference --resume
```

A pause is a durable decision, not a runtime flag: it is written to a database rather than
to the unit, because a restart is exactly the event that must not lift it. Inference goes
in the instance's admission ledger, which is the one place every enrolled profile's queue
can see; delivery is a per-profile outbox, so it is held on each profile's own archive.
Both are honoured by the gate when it admits work, and reported by `status` as `paused`
with the actor and reason. Starting a service never lifts one, `--resume` is the only way
back, and every hold is attributed — with no owner principal configured and no `--actor`,
a pause is refused rather than recorded against nobody.

## Working the archive

```sh
hermes-memory form                       # the bounded list, and nothing else
hermes-memory form --limit 40 --max-jobs 5
hermes-memory form --actor "$USER" --review <digest the list printed>
hermes-memory form --hermes-home ~/.hermes/profiles/work --review <digest>
```

`form` is the whole of formation: it selects records that have no verified projection at
this store's current epoch, charges the instance's shared daily budget, and dispatches
exactly the approved list — one job per record, keyed on that record's own revision. It is
run when an operator runs it. There is no daemon, no cron entry and no service that does
this by itself, and `status`, `doctor` and `capabilities` all say so
(`formation_unattended: false`) rather than letting an empty queue look like a settled one.

The approval is a digest of the list, not of a flag: selected record ids, the route facts,
the processor fingerprint, the token ceiling and `blocking` are all inside it. A hold set
after the list was shown therefore invalidates the approval rather than being run through.

`HERMES_MEMORY_BACKGROUND_BUDGET_TOKENS` is the daily ceiling for the whole machine, shared
across every enrolled profile, because one day of spending is a fact about one GPU. The plan
sizes itself against what the day has left, and when the remaining headroom cannot pay for
even one dispatch it says so as a blocking line — which an `apply` then refuses rather than
shrugs off.

## The worker and its ledger

The backend's own poller dispatches the tasks Hindsight queued. Ours is the process that
does it, so that every spend on the machine's one model is attributable:

```sh
python -m hermes_memory.backend.worker_launcher --check   # run in the backend environment
```

It answers the revision contract against the pinned `0.10.1`, the slot contract (one native
slot, no native reservations, no batch retain, an explicit output cap on every generation
path) and the import surface it composes against, and exits `2` naming each violation. The
unit starts only this module, never the distribution's own script.

Before the engine is asked to run a task, the launcher records the operation — its identity,
the bank it belongs to, the resource it will be charged to — in the instance's admission
ledger. A retry increments the same row rather than becoming a second piece of work, a
cancellation an operator requested is honoured before anything runs, and a lost connection
leaves the operation `uncertain`, never `finished`.

An operation that could not be placed on a route at all is counted as `unattributed`, makes
`status`'s `resource_gate` report **degraded**, and the doctor's remedy points at the
revision contract rather than at a device nothing is holding. The alternative reading — that
somebody else's allowance paid for it — is the thing §8.1.2 exists to forbid.

## Connectors and imports

```sh
hermes-memory sources list --gaps
hermes-memory import --source whatsapp --path ~/exports/phone --dry-run
hermes-memory import --source whatsapp --path ~/exports/phone --policy local-only
```

`sources list` reports each connector's own account: the cursor it stopped at, its
generation and policy version, what coverage it claims, whether its lease is held, and with
`--gaps` the ranges it could not hand over. Coverage is what the source says it has; a gap
is what it could not give, and neither is a claim about what is missing. A source that was
disabled is not forgotten — disabling revokes authorisation for what comes next.

`import --dry-run` answers whether the export can be read and reads nothing else. A real
import needs the scope declared for a connector new to this store (`--policy`), and the
archive it lands in is the one that owns the memory, chosen by `--profile`.

## Explaining and auditing

```sh
hermes-memory explain --record rec_7c2…
hermes-memory explain --artifact art_1f…
hermes-memory explain --goal goal_9a…
hermes-memory explain --not-told "the lease" --at 2026-09-25T23:10:00+01:00
hermes-memory audit --action erasure_confirm --limit 20
hermes-memory audit --source gmail
```

`explain` answers why an item came back, why a goal is due, and why a topic did *not*
interrupt. Raw payloads and goal text stay out of the output unless `--include-private` is
named, and even then secrets are redacted. `audit` reads the ledger of what already happened
and can switch to one connector's own history.

## Backups, restores, and taking a machine away

```sh
hermes-memory backup --reason "before the switch"
hermes-memory backup --list
hermes-memory restore --snapshot snap_27c7…                 # what would be destroyed
hermes-memory restore --snapshot snap_27c7… --review <digest> --actor "$USER"
```

`backup` copies every enrolled profile's store unless one is named, because "this
installation is now recoverable" is false if two of three were copied. Each snapshot
verifies on the way in — file digest, SQLite's own integrity pass, foreign-key sweep, and
the epoch and journal position the manifest recorded.

A restore is gated by a digest of the reading: the snapshot's facts, the notes, and the
**live epoch**. So a capture that lands while the operator is reading invalidates the
approval, and the answer is a refusal plus `hermes-memory stop` first, never a rollback of a
store nobody looked at. What comes back is re-forgotten before it answers a single read:
evidence forgotten after the snapshot stays forgotten, which is the whole point of keeping
the erasure ledger outside the rolled-back file. The store that was there is copied aside
first, so a wrong restore is itself reversible.

```sh
hermes-memory uninstall --keep-data --hermes-home ~/.hermes/profiles/work \
    --actor "$USER" --review <digest>
```

Removing an installation removes the plugin registration, the owned units, this
installation's pointers and the profile binding. It does not delete memory: `--keep-data` is
required rather than defaulted, and a call that asks for anything else is refused by name —
data-preserving is the only uninstall there is. Purging evidence is a separate owner
decision with a displayed manifest (`owner --confirm-forgetting`), never something bundled
with removing a package. The prior provider selection is restored only if the current one
still equals what setup wrote, so an owner who chose something else in the meantime keeps
their choice.

## Upgrades

```sh
hermes-memory upgrade --version /srv/releases/hermes-memory/0.2.0
```

This command is read-only, and there is no flag on it that performs a switch. It inventories
the running and target revisions, the dirty plugin files, in-flight jobs, schema
compatibility, disk and recent backups, and says what a switch would require: pausing
formation and delivery, reconciling active jobs, a coherent backup, migrations rehearsed on
an isolated restored copy, and a quiesced final cutover. Doing it from a terminal in one
command would be a claim about downtime and rollback this build has not measured, so the
plan is printed and nothing else happens.

## What this build does not claim

- **Nothing drains the queue unattended.** Formation is `form`; the worker unit runs the
  backend's own poller for work already queued. `formation_unattended: false`.
- **Guarded delivery is not available yet.** The host facts report
  `guarded_delivery_supported: false` until the host change (H1, §9.5) lands, because a
  pre-dispatch revocation check cannot be performed by a framework that is not consulted at
  the moment of sending. Notifications therefore remain notify-and-draft.
- **The live composition is unverified.** Every launcher test runs against a stand-in
  backend with the pinned shape. Proving the composition against a real `0.10.1` install is
  the P0 gate, and until then no claim here is measured rather than asserted.
- **No false observation readiness.** With consolidation off, `observations` reports what
  coverage actually exists; a semantic layer that was never formed is said to be absent.
- **The DSN is never printed.** The backend's resolved PostgreSQL connection string goes to
  the worker process privately; no command, status stage or finding echoes it, and detail
  strings are redacted on the way out.
