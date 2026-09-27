# Quality: what this build measures before it claims anything

`evals/run_synthetic.py` is the release gate for the framework's *behaviour* — the §12.4
metric table, run over the §12.3 synthetic corpus, with no model and no network. It is one of
the release artefacts §15 asks for: the feature-quality report.

```sh
uv run python evals/run_synthetic.py                 # the report, on a terminal
uv run python evals/run_synthetic.py --json out.json # the same answer, machine-shaped
```

Exit status is the gate: zero only when every *measured* row passed. Rows this build cannot
measure are printed, counted and named with the thing they wait on. They do not pass and they
do not fail, because the honest answer to "did extraction quality hold?" on a machine with
inference switched off is "that was not measured here" — and a report that silently omitted the
row would read as a pass.

## The two shapes

The question set is `evals/cases/checks.py`, and every row it returns is one of exactly two
records:

* **measured** — a value, the criterion it is held to, and the denominator it was computed
  over. The denominator travels with the value: "recall 1.00" of 1 question and of 240
  questions are the same number and different claims.
* **not measured** — the reason, and what has to exist before the row can be scored.

A row is never a third thing: not an estimate, not a placeholder, not a number carried over
from a run that had a backend.

## What the corpus is

`evals/fixtures/corpus.py` writes the fixtures rather than shipping hundreds of hand-authored files:
19 categories from §12.3 (corrections, ambiguous dates, same-name people, quoted duplicates,
tail evidence in a long document, measurements with gaps, unknown answers, claims that cannot
both be true, and the rest) over 12 people, plus held-out questions and 100 eligible / 100
matched no-action proactive scenarios.

Three rules hold it honest:

* Every record carries a code that appears in no other record, so recall is a fact about
  retrieval rather than a scorer's tolerance for near misses, and an absent code has exactly
  one correct answer.
* Event time and arrival time deliberately differ, and some records have no event time at all.
  A corpus where the two coincide cannot tell temporal correctness from luck.
* Nothing is random. Two runs over the same tree produce the same numbers, or a regression is
  indistinguishable from a dice roll.

The plan's `evals/fixtures/` and `evals/cases/` are that generation and that question set, one
module each: a corpus that is *derived* cannot quietly drift from its own claims, because
`check_corpus` reads the claims out of the generator and fails when they disagree.

## What each metric row gets measured

| §12.4 row | Measured here | Waits on |
|---|---|---|
| Privacy/lifecycle | no record reaches a caller outside its scope; no lesson learned from one person's evidence is taught to another — neither by asking for their own, nor by handing the broker a lesson it should not have; forgotten evidence cannot be published (an outbox artifact that quoted it is suppressed); an agent-role caller cannot open an owner door | — |
| Capture/sync | a replayed page forks no second record; one logical event, one journal row; every acknowledged record still readable | crash windows already covered by the contract suite |
| Extraction | — | the P0 authorization to make one bounded live retain: recall, attribution precision, negation handling and schema success are properties of what a model forms |
| Retrieval | required-evidence recall, answer support, temporal correctness, abstention, and "a corrected answer is not returned as if it still stood"; ablation with the lexical index removed | the raw-facts, observations and summaries ablations, which need the pinned backend |
| Proactivity | proposed-notification precision, matched no-action false-interruption rate, recall, timing, and that every quiet case is quiet *for the mechanism it names* | — |
| Compute | a doctor run and a status read reach no model; one flight at a time per physical resource; a pause dispatches nothing; a spent budget refuses before the work; a measurement window reads exactly the samples inside it and cites only them; a series that changed units refuses to be averaged; a sample with no time is counted rather than invented; a withdrawn sample leaves the mean | — |
| Context latency | warm local-context p95, cache hits versus cold reads, deep-read ceiling | the host-side prefetch bound, which needs a real Hermes turn to be late for |
| Formation latency | — | the same live retain; the queue, leases and budgets are scored under Compute |
| Operations | no failing check reported without a remedy, no finding without a reason, a paused stage says why, an owner-only decision is named as waiting, a background pass that stopped is said by both readings, an unanswered cancellation is filed as an intent and never as a stop | — |
| Install | the eleven steps; a clean setup completes; a second setup reports every writing step as done and rewrites no unit or template; the owner's model configuration survives; nothing is ingested by installing | — |
| Recovery | a snapshot verifies; erasures taken after it are reapplied on restore; an erasure of evidence the snapshot never held is carried rather than crashing the restore; the restored store reports itself consistent | — |

The proactivity row is the one most easily satisfied by giving up: a gate that never speaks
scores 100% precision and zero interruptions. That is why the report also prints **recall** and
scores each matched no-action case against the mechanism that was supposed to silence it.
`tests/integration/test_evals.py` forces every answer silent and requires the recall row to
change colour.

## Reading the output

```
proactivity  [5/5 measured]
   ok   proposed notifications are the eligible ones = 1 (of 100) >= 0.90
   ok   matched no-action cases stay quiet = 0 (of 100) <= 0.05
   ...
extraction  [0/0 measured, 2 not measured]
   --   evidence recall, attribution precision, negation handling and schema success
        waits on: the P0 gate — one authorized, bounded live retain against the pinned engine
```

`--quiet` prints the summary line only. A report written with `--json` carries the whole
per-check `detail`: the categories that were sparse, the questions that were missed, the
latency distributions, the stages that refused and why.

The scratch installation is rebuilt from nothing on every run. `--root DIR` names where, and
the command **refuses** a non-empty directory rather than clearing it: measuring over a
previous run's store would report that run's tombstones, and `rm -rf` on a path somebody else
typed is not this command's decision to make. On a tmpfs the whole run takes a couple of
seconds; on disk, SQLite's durability work dominates it (the ingest of 252 records is the same
few milliseconds either way).

## The rule the harness itself is held to

A check that cannot fail is not measuring anything, so each gate has a case in
`tests/integration/test_evals.py` that breaks the property upstream — duplicate a code,
drop a category,
put an answer where the corpus promises none, force the gate silent, mislabel a matched
scenario — and requires the row to go red. When one of those cases is missing, the mutation
driver over `evals/` and the recovery path says so; the fix is the missing test, never a
softer assertion.

## Where a test belongs

The suite is split by what a failure would mean, not by which module it imports:

| Directory | Holds |
|---|---|
| `tests/unit/` | one component's rules, in-process, over a temporary store — the storage invariants, the comparison rules, the state machines, the budgets |
| `tests/contracts/` | a shape something else consumes: an adapter's paging and its malformed input, the pinned backend's bridge, the loopback gate's HTTP surface, the served process's bind |
| `tests/integration/` | a door wired through several components at once — the operator CLI, `status`, the explanation of a verdict, the maintenance pass, the harness itself |
| `tests/hermes/` | the host side: the provider plugin, the two entry points that reach outside the machine, the conversation export |
| `tests/installer/` | anything that would touch a running installation — the setup transaction, upgrade and uninstall plans, unit files, profile enrollment, the inventory, the compatibility manifest, `doctor` |
| `tests/faults/` | the failure paths on their own terms: cancellation, snapshots and the crash window between restore and restart |

`tests/conftest.py` and the shared helpers (`connector_script.py`, `learning_cases.py`,
`plugin_loader.py`) stay at the root, and `pythonpath = ["tests"]` keeps them importable from
every directory. A shared *fixture* is a helper module rather than an import from a sibling
test file: the case data the evaluation door and the learning rules both answer to is one
object, so neither can drift from it.

The plan's `tests/migration/` is absent on purpose. Migration was retired in favour of a
fresh installation, and an empty directory claiming coverage of a retired contract is the
same lie as a test that asserts nothing.
