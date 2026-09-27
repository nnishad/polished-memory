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

The report also carries four things an operator reads next:

- `erasure_backlog` — obligations that are not verified gone everywhere, tombstones, and how
  many intents are `awaiting_owner`.
- `awaiting_owner` — the identity candidates, candidate assertions and erasure intents that
  only a human may decide. When any exist, a note names the command that can decide them.
- `queue` — the age of the oldest waiting job against the allowance, with
  `"formation_unattended": false` beside it. A queue that is not being drained by anything
  is said as that: *only `hermes-memory form` works this queue — nothing drains it by
  itself*.
- `background_pass` — whether the zero-inference pass is scheduled at all, when it last
  wrote down that it ran, and whether reminders are waiting behind a loop that is not
  running. It reads the pass's own record rather than trusting that a thread in another
  process is alive.

Nothing here is a health check that calls a model. `backend` reads the projection ledger,
which is true without a socket; asking whether the backend is answering is the probe below.

## Why it is not

```sh
hermes-memory doctor
hermes-memory doctor --probe             # one request: is anything answering
hermes-memory doctor --synthetic-probe   # one bounded synthetic retain and recall
```

Sixteen checks run by default: layout, database, schema, configuration, coverage, queue,
the background pass, provenance, lineage, erasure, delivery, gate, a credential scan over the
newest records, leases, the projection ledger, and the release manifest. `--probe` adds a
connectivity request and
`--synthetic-probe` a bounded round trip in a bank of its own; the report says which probes
ran, so a later reader can tell what was actually looked at. Exit is `0`, or `1` when a check
reached `fail`; refusals (an unreadable configuration, an unenrolled profile) exit `2` with
the reason on stderr.

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
hermes-memory owner --activate-lesson chase-invoice@3 --reason "this is how we work"
hermes-memory owner --retract-lesson chase-invoice@3 --reason "wrong for this client"
hermes-memory owner --confirm-lesson chase-invoice@3 --reason "it worked on two accounts" \
    --evidence rec_2f9…
hermes-memory owner --contradict-lesson chase-invoice@3 --reason "it cost the client" \
    --evidence rec_8b1…
```

A proposed habit (`lesson`) is the fourth kind of thing an archive must not approve for
itself. Activation is the owner's, or a passing evaluation's — never the proposer's, and the
store refuses a run whose runner is the account that filed the candidate. `--version` (or
`name@N`) is required for everything except a retraction, because the versions of one lesson
disagree with each other by construction, and withdrawing the newest is a different act from
withdrawing the one that was being taught.

A lesson is retrieved into a turn only while it still holds up: forgotten evidence, a
withdrawn evaluation or more checked failures than successes retire it on the next read, not
on a review nobody schedules. `--list` names those under `lessons_for_review`, and
`status` counts them among the owner-only decisions.

`--list` shows what awaits, per enrolled memory, **counted rather than quoted**: intent and
candidate ids, who proposed them, the digest that signs a preview, and the size of the blast
radius — not the record identifiers or texts behind them, which are read with `explain` by
somebody allowed to see them. A proposed lesson is the exception and is quoted in full: the
sentence *is* the decision, and nobody can approve a rule they cannot read. The listing opens
every store read-only; a command run in order to look must not be the one that migrates it.

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
hermes-memory pause --scope capture --source gmail --actor "$USER" --reason "under review"
hermes-memory pause --scope capture --source gmail --resume
```

A pause is a durable decision, not a runtime flag: it is written to a database rather than
to the unit, because a restart is exactly the event that must not lift it. Inference goes
in the instance's admission ledger, which is the one place every enrolled profile's queue
can see; delivery is a per-profile outbox, so it is held on each profile's own archive.
Capture is narrower than either: it names one connector, and stops the reading of it and
nothing else — the evidence already stored stays formable, and what was already held from
formation stays held. A source registered in two profiles' memories is held in both,
because a hold that left the second one polling the same mailbox would be a false report.
All three are honoured where work is admitted, and reported by `status` as `paused` with
the actor and reason; `pause` answers with `held`, which is read back from the ledgers
rather than echoed from the command that was just run. The inference hold comes back as
`inference_hold` — actor, reason, policy version and the moment it was written — because a
hold that stops every profile on the machine is the one decision a second operator needs
to be able to attribute, and `instance_hold_by`/`instance_hold_reason` say the same thing
in the status stage itself. Starting a service never lifts one,
`--resume` is the only way back, and every hold is attributed — with no owner principal
configured and no `--actor`, a pause is refused rather than recorded against nobody. A
`--source` on the two installation-wide scopes is refused rather than ignored, and naming a
connector no memory registers is refused too: a typo must not look like a hold.

A held installation starts differently. `hermes-memory start` under an inference hold brings
up the admission unit and asks for nothing else, reporting `not_started` and the door that
lifts the hold: the backend dispatches an embedding probe as it comes up, that probe goes
through the admission gate, and the gate denies a new dispatch for as long as the owner's
hold stands — so asking for the backend would put it into a restart cycle that ends with the
unit `failed` and its bound worker taken down beside it. A machine that looks broken because
it was told to wait is not a machine that is waiting. The worker's own launcher reads the
same row and waits where it can: it prints `{"worker": "waiting"}` with the actor and the
moment, polls, and composes the engine only once nothing is held.

## Settling an admission that never came back

```sh
hermes-memory gate
hermes-memory gate --resolve wait_9f2c1a… --outcome cancelled --reason "cancel --operation "
                   "op-117 was acknowledged by the backend"
```

`gate` reads the instance's admission ledger: which device is held, by whom and for how
long, what is waiting behind it, and what is *unresolved*. The last of those is the reason
the door exists. A request whose connection was lost leaves a reservation that keeps its
device blocked on purpose — nothing this process knows proves the model server stopped, and
handing the slot out again could put a second request on a GPU that is still busy. The
doctor reports that state as degraded and it does not repair itself: `reap_expired` turns an
expired lease into an uncertain reservation, never into a free device.

So `--resolve` is how an operator establishes the answer from outside — a cancellation the
backend acknowledged, a record the reconciler found projected, a model server checked by
hand. It refuses without both `--outcome` and `--reason`, because "impatient" is not a
settlement, and it refuses a reservation its holder already released. Nothing is charged to
the daily budget: the token count of a request nobody saw the end of is not a number worth
adding to a total. The ledger keeps both records — the holder that could not answer, and
the operator who answered for it, with the name given.

## Working the archive

```sh
hermes-memory form                       # the bounded list, and nothing else
hermes-memory form --limit 40 --max-jobs 5
hermes-memory form --actor "$USER" --review <digest the list printed>
hermes-memory form --hermes-home ~/.hermes/profiles/work --review <digest>
hermes-memory form --reconcile           # ask the backend about submissions we stopped
                                         # being able to answer for
```

`form` is the whole of formation: it selects records that have no verified projection at
this store's current epoch, charges the instance's shared daily budget, and dispatches
exactly the approved list — one job per record, keyed on that record's own revision. It is
run when an operator runs it. There is no daemon, no cron entry and no service that does
this by itself, and `status`, `doctor` and `capabilities` all say so
(`formation_unattended: false`) rather than letting an empty queue look like a settled one.

A dispatch is submitted asynchronously and then *followed*: the submission's own answer says
only that the engine accepted the work, and the projection is confirmed when the operation
reaches `completed`. The device slot is renewed each round of waiting rather than assumed, so
a consolidation that takes an hour does not turn into a lapsed lease that some other caller is
promoted into. Each ending is reported as what it was: `failed` and `not_found` retry,
`cancelled` ends the job rather than resubmitting the work somebody stopped, and a wait that
runs out holds the slot as uncertain with the operation identity left on the row — because the
model may still be reading the document, and a free device would be the second lie.

That is what `--reconcile` is for. It sends no model request and charges no budget, so it
needs no approval digest; it asks the backend, one bounded list at a time, what became of the
submissions this machine cannot account for, and writes down the answer it was given. A row it
cannot settle stays outstanding and is asked again next time.

The approval is a digest of the list, not of a flag: selected record ids, the route facts,
the processor fingerprint, the token ceiling and `blocking` are all inside it. A hold set
after the list was shown therefore invalidates the approval rather than being run through.

`HERMES_MEMORY_BACKGROUND_BUDGET_TOKENS` is the daily ceiling for the whole machine, shared
across every enrolled profile, because one day of spending is a fact about one GPU. The plan
sizes itself against what the day has left, and when the remaining headroom cannot pay for
even one dispatch it says so as a blocking line — which an `apply` then refuses rather than
shrugs off.

## The background pass

```sh
hermes-memory maintain                          # one bounded pass, all five sections
hermes-memory maintain --section proactive
hermes-memory maintain --limit 5 --at 2026-09-15T09:00:00+00:00
hermes-memory maintain --hermes-home ~/.hermes/profiles/work
```

Five things were built and had no caller: due reminders were never taken, a summary that a
correction invalidated was reported as stale but never queued for the refresh that would
clear it, an identity candidate citing forgotten evidence stayed pending forever, a job
waiting out a backoff was listed while being unclaimable, and nobody counted what an
erasure still owes a backend. `maintain` is the pass that runs them, and the runtime unit
runs it on a period (`HERMES_MEMORY_MAINTENANCE_INTERVAL_S`, 900 seconds by default, `0` to
hand the timer back to you).

**The pass never spends a model call.** That single rule is why it can run unattended while
`form` cannot: `form` costs tokens, a device slot and a unit of the daily budget, so it
needs the digest of a list somebody read and the name of who approved it. This takes none of
those, and its engine is built without a model client at all, so there is nothing to reach
by mistake. A reminder is kept in the owner's own words — which is why turning inference off
does not turn proactivity off as a side effect.

Because it runs by itself, it must not be able to destroy anything by itself either. A
refusal that means *not now* — shadow mode, an operator pause, a spent budget, a snooze —
leaves the reminder standing, and the pass reports it under `deferred`. A refusal that means
*not at all* — an opted-out topic, a goal the owner finished, an expired promise — closes
the event, and is reported under `suppressed`. Getting those two wrong in either direction is
the failure this section exists to describe: a timer that suppressed on shadow mode would
have eaten every reminder on a new installation before anyone had agreed to be interrupted.

`status` reports the pass as `background_pass` (the last one recorded, its age, what is
waiting) and `doctor` checks it as `background`, because a dead scheduler is indistinguishable
from a machine with nothing to do unless the pass writes down that it ran.

The heartbeat carries a failure too. The sections run in order, so a raise halfway through
would leave nothing recorded and the loop would report exactly like one that stopped — which
is what happened on a live machine whose queue section reaped expired leases against a
read-only admission ledger: the timer kept its period, the heartbeat stopped, and `status`
said "the runtime unit's scheduler is not running" for hours. A pass that raised now records
`failed_section` and `error` in its own heartbeat, `status` says *the scheduler is running but
the pass is not finishing*, and `doctor` raises it as `FAIL` with the command that reproduces
the raise in your own terminal. Reaping is a write, so the pass opens the admission ledger
writable — and, like every reading, invents no ledger where none exists.

## Promises and their conditions

```sh
hermes-memory goal --list                    # what is owed, candidates included
hermes-memory goal --due-now                 # what is due, with each condition answered
hermes-memory goal --id gol_91c… --snooze 2026-09-16T09:00:00+00:00 --reason "after the trip"
hermes-memory goal --id gol_91c… --revise --due "2026-09-20T09:00:00+00:00" --reason "moved"
hermes-memory goal --id gol_91c… --activate --reason "yes, that is mine"
hermes-memory goal --id gol_91c… --complete --reason "called them"
hermes-memory goal --id gol_91c… --history
```

Every transition is owner-gated in the store itself, so `--actor` naming somebody else is
refused rather than recorded; `--actor` defaults to `HERMES_MEMORY_OWNER_PRINCIPAL`, and every
one of them requires a `--reason`, because the decision outlives the conversation that made it.
`--revise` opens a **new revision** rather than editing a due time in place, which is what makes
an in-flight reminder safe to ignore: a handoff against revision 1 describes an event that no
longer describes the goal. A `--snooze` that would end before the goal is due is refused with
"revise the due time instead" — it would suppress nothing and would look like a decision.

A goal proposed from outside the owner — by an agent through the `memory_goal` tool, or by a
background pass — is a **candidate**: it schedules nothing, reminds for nothing, and appears on
`owner --list` and in `status`'s `awaiting_owner.goal_candidates` until the owner adopts it.
`--activate` gives it a clock. A candidate is quoted in the listing rather than counted only,
because the sentence is the decision.

**Conditions are evaluated when the event fires.** A promise with a predicate — *if nobody
replies*, *once the invoice is over 14 days*, *when the thing I am waiting for finishes* — is
not settled by a clock, and eligibility asks the store at the moment of the handoff. Each
answer decides what happens to the event, and the two families are the same distinction the
background pass is built on:

| The condition said | The event | Because |
| --- | --- | --- |
| satisfied | goes on to the policy gate | the premise holds now |
| pending | held over, still due | the instant has not arrived |
| unknown | held over, still due | the store cannot say, and an unanswerable question is not a licence to act |
| failed | suppressed | the premise was checked against current coverage and is not true — this is a durable no |
| no goal ledger wired | not checked here | the answer says which of the two it is |

`goal --due-now` shows the same evaluation from the owner's side, per promise: `ready`, and the
`unknown` list naming the conditions the store could not answer. Held-over events are counted
as `deferred` by the background pass, never as `suppressed`, so a reminder that is merely
waiting is distinguishable from one that is finished.

## Promoting a habit by a run

A lesson is a rule the archive learned about its own procedure. It starts as a candidate
and teaches nothing until either the owner promotes it by name or a run licenses it — and a
run means an evaluator the owner authorized, not the system agreeing with itself:

```console
hermes-memory evaluate --lesson chase-invoice                      # what the archive knows
hermes-memory evaluate --lesson chase-invoice --suite cases.json \
        --model-version remote-9b                                   # run it, and promote on a pass
hermes-memory evaluate --lesson chase-invoice --suite cases.json \
        --model-version remote-9b --no-promote                       # the verdict, without the rule
```

The evaluator is configured, never passed on the command line: `HERMES_MEMORY_EVALUATOR_COMMAND`
names an absolute program, and it is executed with no shell, a scrubbed environment (add
`PATH`-like names through `HERMES_MEMORY_EVALUATOR_ENV` if a fixture genuinely needs one), a
timeout of `HERMES_MEMORY_EVALUATOR_TIMEOUT_S` and a bounded read. It is asked about one
case at a time over a pipe — the lesson and the case, and nothing else — and answers
`{"outcome": "pass"|"fail"|"error"|"skipped", "detail": "..."}`. A timeout, a non-zero exit,
an answer that is not JSON or an outcome word the ledger does not know is recorded as
`error` for that case, which means a broken evaluator cannot manufacture a promotion.

A suite is a JSON file: a list of cases, or `{"cases": [...], "baseline": {...}}`, each case
carrying `case_id` and a `role` of `targeted` or `regression`. The verdict is derived from
the recorded cases — a lesson needs both a case it was supposed to fix and one that already
worked — and it is bound to the lesson text, the fixtures, the baseline, the code version and
the model version. That binding is why `--model-version` is required: a verdict with no
version beside it can never be found stale, and a stale verdict is a different and more honest
answer than a pass that quietly outlived the thing it tested.

Forgetting the evidence a lesson was drawn from ends the run behind it as well: the verdict
goes to `stale` in the same transaction as the tombstones, the lesson stays on file for the
owner to read and withdraw, and `activate_evaluation` refuses it until a fresh run is scored
against what is left.

## Who asked for what, and when

```sh
hermes-memory profiles                          # the map, and the decisions behind it
hermes-memory profiles --profile work           # one profile's decisions only
hermes-memory profiles --review <digest>        # what an approval actually did
```

An enrollment, a move or a retirement is written to the instance ledger with the actor who
attributed it and the review digest they approved. `profiles` lists the current map *and*
that history, and `--review` answers the question an operator asks afterwards: given a
digest, what was applied under it. A digest nobody approved is answered as `applied: null`
rather than as an error, because "did this happen?" is a legitimate question about a number
that was printed on a screen.

## Readings of a scope

A summary is a claim about a window of evidence, and it stays in its own table so it can
never be cited by the next summary. Three things have to agree for that to be worth
having: something writes them, something reads them, and something takes them out of
circulation when the evidence under them goes.

**Writing one** is an operator act, because it is a model call over private evidence:

```console
hermes-memory summarize --scope project:survey          # lists the window, asks nothing
hermes-memory summarize --scope project:survey --review <digest> --actor jugaadu
```

The scope is what the summary will be a claim *about*, and it is resolved locally:
`project:<name>`, `thread:<name>` and `account:<id>` read the metadata the adapters
carried, `source:<name>` takes a whole connector, and `day:<date>` / `week:<date>` take
the evidence that *occurred* in that span (not the evidence that arrived in it). The
listed window is what the digest covers, so evidence captured between the two commands is
a different pass and the second one refuses. It also refuses when the scope holds more
records than `--limit` allows: narrowing is the operator's call, not a prefix that gets
presented as the whole. A `mental_model` reaches past the evidence, so only the owner
principal may write one.

**Reading them** happens in every context packet. The broker takes the scopes the
authorized evidence already belongs to and quotes the current reading of each — labelled
as a reading, with a caveat saying it is not a second sighting of the evidence. A summary
whose evidence no longer stands is absent from the packet rather than replaced by the
older one that still is on file, and readings are the last thing dropped when the token
ceiling bites.

**Withdrawing them** is automatic. Confirming an erasure withdraws every published
summary that quoted the forgotten evidence, in the same transaction as the tombstones, and
books a refresh of that scope: the reading is not merely gone, the scope is owed a new
one. `maintain` books the same promises when a correction makes a window stale, and
`summarize` answers every promise its window reaches.

## Stopping work

```sh
hermes-memory cancel --job job_3f9a… --reason "wrong bank"
hermes-memory cancel --operation op-117 --reason "the owner asked"
hermes-memory cancel --list
hermes-memory cancel --job job_3f9a… --reason "superseded" --actor worker-2
hermes-memory cancel --operation op-117 --reason "…" --hermes-home ~/.hermes/profiles/work
```

Two halves, and only one of them is ours. The queued job is ours outright: cancelling it
closes the row, releases its lease and keeps `claim()` from ever handing it out again. The
backend operation is ours to *write down* and not ours to finish. So the intent is recorded
before anything is called — a crash between the two leaves "somebody asked this to stop and
nobody knows more", never a silently un-cancellable operation — and what the report says next
depends entirely on what came back:

| The backend said | The ledger says | Why that is the honest word |
| --- | --- | --- |
| an acknowledgement | `cancelled`, intent `confirmed` | both halves agree, and the charge the worker reported stays |
| nothing, because the socket dropped | `uncertain`, intent `requested` | it may still be running upstream; `cancelled` would be a wish |
| `HTTP 4xx/5xx` | unchanged, intent `requested` | it refused, so nothing about the operation is claimed |
| no such endpoint at this revision | unchanged, intent `requested` | a fact about the backend's shape, not a failure to cancel |
| nothing, because no route is configured | unchanged, intent `requested` | nobody was told; the intent still stops the *next* dispatch |

An operation the ledger already settled is never asked again, and never credited to this
request: an operation that `finished` while the cancellation was in flight is reported as
`finished` with the acknowledgement beside it, because writing `cancelled` over it would make
the record claim a stop that did not happen. Which states count as settled stays the ledger's
own business rather than a list copied out of it.

`--list` is a reading: it opens the admission ledger read-only, creates nothing and migrates
nothing, and answers `absent` on an installation that has never dispatched. It is the queue an
operator works from after a restart, because the intent outlives the process that wrote it.
`status` reports the same rows as `operations.cancellations_owed` and holds the gate stage at
**degraded** while any are unanswered; `doctor`'s `gate` finding names `cancel --list` as the
remedy. A job that has no backend operation behind it cancels happily with no ledger at all —
closing the queue row is the whole of what can be done there.

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
hermes-memory sources reconfigure --source gmail --policy private-api \
    --actor "$USER" --reason "the account moved"
hermes-memory import --source whatsapp --path ~/exports/phone --dry-run
hermes-memory import --source whatsapp --path ~/exports/phone --policy local-only
```

`sources list` reports each connector's own account: the cursor it stopped at, its
generation and policy version, what coverage it claims, whether its lease is held, and with
`--gaps` the ranges it could not hand over. Coverage is what the source says it has; a gap
is what it could not give, and neither is a claim about what is missing. A source that was
disabled is not forgotten — disabling revokes authorisation for what comes next.

`sources reconfigure` is what an operator runs when a connector's old position no longer
describes the same stream: a re-mapped account, a changed endpoint, a narrowed or widened
scope. It starts a new generation, which strands any writer still holding the previous
fence instead of committing under the new declaration, drops the cursor so the next read
begins at the start, resets coverage to `unknown`, and re-records the ingestion scope.
Run without `--review` it writes nothing and prints the connectors it found, what each one
currently holds and `review_digest`; the write happens only against that digest, which is
computed over the source, the scope, the actor, the reason and the connectors that were
shown. A digest from a different plan — another source, another scope, another attribution
— is refused, so re-running a preview with one word changed approves nothing until the new
preview is read. The archive is not touched: records already stored stay stored, and a
re-read that arrives again is refused as the duplicate it is. `--reason` is required,
because the full re-read of a source that may hold other people's messages is felt by
whoever reads the ledger next.

`import --dry-run` answers whether the export can be read and reads nothing else. A real
import needs the scope declared for a connector new to this store (`--policy`), and the
archive it lands in is the one that owns the memory, chosen by `--profile`.

A sample fixture (`.csv`, `.tsv`, `.jsonl`) has a second decision, and `import` refuses to
make it for you: `--granularity sample` keeps every row the source wrote, which is what
`hermes-memory measure` reads later; `--granularity series` collapses each
device+measure+unit group into one described summary, which is smaller and can never be
re-windowed — you cannot ask a mean for a month you did not keep the rows of. Prose
readers (`files`, `email`, `whatsapp`) have one shape per item and refuse the flag.

## Reading a measurement

```sh
hermes-memory measure --list
hermes-memory measure --what weight --unit kg \
    --since 2026-03-01T00:00:00+00:00 --until 2026-03-31T23:59:59+00:00
```

A measurement is a number that says what it measured, in what unit, from which device and
when; without those four it is not reusable, so the reading carries them and the
arithmetic is done in this process rather than by a model. The answer is the statistic
*and* its accounting: how many samples were placed in time, how many carried no time at
all and were dropped by the window (`unplaced_excluded`), how many named a different unit
and were left out (`units_seen`), and the record ids of exactly the rows that were
averaged. A series whose stored samples disagree about units returns `statistics: null`
and names both units — a mean across kg and lb is a number that only looks like a
measurement. `--list` answers "what can I ask?" before anything is asked, and refuses to
be combined with a question. Bounds have to carry a timezone: `2026-03-01` sorts below
every timestamp of that day and would quietly drop the samples it was written to keep.

## Whose memory is being read

`status`, `doctor`, `measure`, `sources list`, `audit` and `explain` all answer about one
memory. With a single profile enrolled that is unambiguous and they use it; with two or
more they refuse until `--hermes-home` (or `--profile`, where the door has one) names
which, because a reading about one person that quietly came out of another person's
archive is the failure the profile ledger exists to prevent. `init` takes the same
selector: enrollment maps a profile to a memory, and `init --hermes-home <home>` is what
opens that memory's store and blob directory beside it.

## Explaining and auditing

```sh
hermes-memory explain --record rec_7c2…
hermes-memory explain --artifact art_1f…
hermes-memory explain --goal goal_9a…
hermes-memory explain --not-told "the lease" --at 2026-09-25T23:10:00+01:00
hermes-memory audit --action erasure_confirm --limit 20
hermes-memory explain --lesson chase-invoice
hermes-memory explain --summary sum_4a1c…
hermes-memory audit --source gmail
hermes-memory audit --actions
hermes-memory audit --actors
hermes-memory audit --decisions erasure
hermes-memory audit --timeline rec_7c2… --include-private
```

`explain` answers why an item came back, why a goal is due, and why a topic did *not*
interrupt. Raw payloads and goal text stay out of the output unless `--include-private` is
named, and even then secrets are redacted. `audit` reads the ledger of what already happened
and can switch to one connector's own history.

`--actions` counts the kinds of thing that have happened, `--actors` counts who was acting,
`--decisions` answers the question an auditor asks first — which of these were owner-only
decisions, and who made them — and `--timeline` assembles everything this store can say
about one record. A timeline leaves the record's own text out unless `--include-private`
names it, and even then secrets are redacted.

## Backups, restores, and taking a machine away

```sh
hermes-memory backup --reason "before the switch"
hermes-memory backup --list
hermes-memory backup --keep 8        # retain eight snapshots per profile, newest first
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

## Which door carries which operation (§6.7)

The plan's §6.7 table names eighteen operations with a caller and a contract each. Every one
of them is implemented and reachable, but the framework owns no HTTP listener of its own: the
only socket this build binds is the model-admission gate (§8.1), whose `/v1/chat/completions`
and `/v1/embeddings` are the *upstreams'* routes, not these. The operations are carried by the
surfaces that have callers today — the Hermes plugin, which reaches the library in-process
under the calling activity's own profile scope, and the CLI, which is where an owner or
operator stands. Each row's contract (bounded payload, request identity where replay matters,
revision/epoch precondition, the caller's authority) is enforced in the library the doors
share, so no surface can talk to a memory it was not bound to.

| §6.7 operation | Carried by | Authority |
|---|---|---|
| `GET /v1/status` | `hermes-memory status`; provider `memory_status` | scoped agent, operator |
| `POST /v1/context` | provider prefetch and `ContextBroker` (per-channel degradation, deadline, token budget) | scoped agent |
| `POST /v1/remember` | provider `memory_remember` → canonical receipt; `captured` reported apart from `formed` | scoped agent |
| `POST /v1/ingestion/pages` | connector runtime and `import`; fenced page commit against lease/generation/epoch | source connector |
| `POST /v1/capture/events` | the plugin's durable capture spool (event id, checkpoint, settle) | profile plugin |
| `GET /v1/evidence/{id}` | `audit --timeline`, `explain --record`, citations in `memory_recall` | scoped reader |
| `GET /v1/explanations/{id}` | `explain` for a packet, artifact, goal, lesson, summary or quiet topic | scoped reader |
| `POST /v1/assertions` | assertion store through the provider and `owner --confirm-assertion` | scoped agent proposes, owner confirms |
| `POST /v1/identity-candidates` | provider `memory_identity_candidate` — always unconfirmed | scoped agent |
| `POST /v1/owner/identity-decisions` | `owner --confirm-identity / --reject-identity / --revoke-edge` | owner only |
| `POST /v1/goals` and transitions | provider `memory_goal`; `goal --activate / --complete / --cancel / --snooze / --revise` | scoped agent proposes, owner settles |
| `POST /v1/feedback` | outcome log behind `owner --confirm-lesson / --contradict-lesson / --retract-lesson` | owner, with the receipt it cites |
| `POST /v1/forget-requests` | provider `memory_forget_request` — preview and impact manifest, erases nothing | scoped agent |
| `POST /v1/owner/forget-confirmations` | `owner --confirm-forgetting --digest …` — the fence plus a durable obligation | owner only |
| `POST /v1/delivery/claim` | the delivery bridge: one artifact per claim, lease and attempt correlation | fixed bridge |
| `POST /v1/delivery/receipts` | the same bridge's proof path; a caller cannot self-report delivery | trusted receipt bridge |
| `POST /v1/owner/controls` | `pause` / `start` / `stop`, with actor and policy version recorded | owner only |
| `GET /v1/jobs/{id}` | `form`, `maintain`, `cancel --job`, and the queue stage of `status` | scoped agent, operator |

The four owner rows are the reason there is no framework HTTP surface on the agent's path: an
endpoint that accepted an agent-role credential could be asked to confirm a forgetting or
revoke an identity, and the plan requires those to be unreachable that way. A socket would add
a place to put private evidence, a credential to steal and a log to leak from, without adding a
caller that does not already have an in-process, profile-scoped one. If a non-Hermes client
ever needs these, the route table belongs beside these same library calls — the authority rule
is in the stores and the broker, not in whichever transport reaches them.

## What this build does not claim

- **Nothing drains the queue unattended.** Formation is `form`; the worker unit runs the
  backend's own poller for work already queued. `formation_unattended: false`. The
  background pass is the opposite kind of thing, and the distinction is the point: it runs
  by itself precisely because it dispatches no inference — it reports the queue's age and
  what is ready to retry, and claims none of it.
- **The background pass does not deliver anything.** It prepares artifacts into the outbox.
  A transport claiming them is the host's decision, and the queue sitting unread is a normal
  state, reported as one.
- **Guarded delivery is not available yet.** The host facts report
  `guarded_delivery_supported: false` until the host change (H1, §9.5) lands, because a
  pre-dispatch revocation check cannot be performed by a framework that is not consulted at
  the moment of sending. Notifications therefore remain notify-and-draft.
- **The live composition is unverified.** Every launcher test runs against a stand-in
  backend with the pinned shape. Proving the composition against a real `0.10.1` install is
  the P0 gate, and until then no claim here is measured rather than asserted.
- **No false observation readiness.** With consolidation off, `observations` reports what
  coverage actually exists; a semantic layer that was never formed is said to be absent.
- **No shipped adapter reads attachment bytes.** The store honours them: an envelope that
  brings `data` has it content-addressed, chunked and written in the *same* transaction as
  its record, so there is no moment in which a record names a file the database does not
  hold; the ingestion receipt keeps the name, type, size and hash and never the bytes; and
  replaying the same revision restores a blob lost between the two writes. What no adapter
  does is read somebody's file to supply those bytes, because that is an authorization none
  of them carries — the email connector deliberately records `ingested: false` beside each
  name it saw, so the honest statement of the state is "names held, bytes not held unless
  they were given".
- **The DSN is never printed.** The backend's resolved PostgreSQL connection string goes to
  the worker process privately; no command, status stage or finding echoes it, and detail
  strings are redacted on the way out.

## Capabilities without a door

Four things in this build are complete, tested and unreachable from any command. Each is
here because the plan requires the *behaviour* and no shipped actor has cause to invoke it
yet; none is a gap waiting for a fix, and none should be mistaken for a claim that the
reading above is complete.

- **The change journal has no downstream consumer.** Only committed changes are journaled,
  and every consumer checkpoints and replays independently (`SyncController.changes` /
  `advance`), so one stalled consumer never holds the upstream cursor or another consumer
  back. Nothing needs that today: projections are claimed transactionally when their record
  commits, not by following the journal. The consumer checkpoints are still durable and
  still survive a restore, because a cursor that restarts at zero would re-emit delivered
  changes.
- **Structural topics have no writer.** `IdentityStore.link_topic` keeps a source-supplied
  topic apart from a model-generated label, and `accounts_for_topic` answers only for the
  structural kind — the separation §C7 requires is enforced where the link is made. No
  shipped adapter supplies topics at all, so the table is empty on a real installation and
  nothing is scoped by it.
- **The erasure ledger can be merged and re-applied by hand.** `ErasureManager.absorb` keeps
  an original `requested_at` through a merge (a restored node must not look like it forgot
  on the day it was restored) and `reapply` re-forgets what a snapshot brought back. `restore`
  does both inline, atomically, before the store is readable — which is the only way either
  should be reached in ordinary operation. There is no command because a hand-merged ledger
  is not a routine act.
- **The client maps routes the product does not yet call.** `HindsightClient.stats` stands
  behind the same pinned capability table as the routes in use, so a contract is stated in
  one place: `stats` is a coverage *basis*, never proof. A stage that needs a capability
  asks for it by name and refuses to work when the running build does not route it.
  `retain_async` and `operation` no longer belong on this list: a formation pass submits its
  work under the durable submission identity and then waits for the operation, and
  `hermes-memory form --reconcile` is the door that asks the backend about a submission this
  machine stopped waiting for.

`hermes-memory explain` and the audit trail remain the way to see what *was* done: a
capability with no door is also a capability that leaves no records.
