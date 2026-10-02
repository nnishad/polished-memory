# Live evaluation isolation incident

Date: 2026-10-02. Status: original test stopped; owner authorized continued repairs and confirmed all existing data is disposable test data. A schema-compatible release has now been activated, health checked, and exercised with an isolated three-case retain/recall smoke run. Original incident-operation metadata has not been fully reconciled. **The full implementation plan is not complete.**

## Authorized scope and isolation failure

The owner authorized isolated synthetic model and memory-bank tests, not personal-message sampling, production migrations, installed-runtime changes or Telegram delivery. Earlier synthesis smoke tests used only explicit synthetic text through the owned admission endpoint.

The new round-trip runner incorrectly used `dataclasses.replace(settings, data_dir=scratch, bank_id=test_bank)`. `Settings.db_path` and `Settings.blob_dir` are independent fields, not properties derived dynamically from `data_dir`. The runner therefore opened the production canonical database despite creating a scratch directory. This was an implementation error, not an authorized expansion of scope.

## Confirmed impact

- Database touched: `/home/jugaadu/data/hermes-memory/data/canonical.db`.
- Opening the current source EvidenceStore applied source migrations **0015_durable_fences, 0016_retain_payload_contract and 0017_projection_generations** to the previously schema-14 database. No downgrade or rollback has been attempted. These migrations include lifecycle fence/backfill changes, not just version-marker updates; consult their source before planning repair.
- Three synthetic canonical records were inserted:
  - `rec_c826ea597ee7ab34030500be09520376` — source `synthetic`, source ID `roman-hinglish`.
  - `rec_eb55eb8739ea4a40f7c302cb87c30052` — source `synthetic`, source ID `devanagari-hindi`.
  - `rec_bfb3d4eb52be28ae9e3430b36104e13a` — source `synthetic`, source ID `english`.
- Formation selected existing canonical evidence, not just those synthetic records. One existing record, `rec_2d647e2232f7271067b729d5cdc30905`, was submitted to the newly created bank `eval-4e5a8311dc4e45fe8db0e7ae413533e4`.
- Backend document: `hdoceb4c3504e99d9038ebc6f14e09f7d048`. Operation: `e0623602-15d7-4440-ad7a-e2f892435ee3`.
- Formation queue entries and a submitted mapping were written in the production database. These remain for audited reconciliation; they have not been silently deleted or marked successful.
- The operation failed extraction. The installed backend hop returned a schema incompatibility error: it does not recognize migration `0015_durable_fences`. This demonstrates an installed-runtime compatibility problem after the unintended migration; a production health recovery is not claimed.
- The submitted document is verified **absent**. The test bank reports **0 nodes, 0 links, 0 documents and 0 observations**, but also **1 pending operation**, 2 failed operations and 1 cancelled operation. This is not proof that all operation payload/history metadata is gone. The bank shell/operation records have not been deleted.
- No Telegram message or installed service change was performed by this test. The existing canonical record itself was not deliberately edited or erased. A full before/after production-data comparison has not been performed.

## Containment

The evaluation process was stopped with SIGINT after the failure was identified. The first sandboxed signal could not reach the host process; an explicit approved escalation stopped the exact test PID. A subsequent process lookup confirmed the runner was no longer active. The runner's cleanup path ran; a read-only backend check confirmed the failed operation and absent document described above.

No further live tests are authorized by assumption after this incident. No production SQL cleanup, schema downgrade, snapshot restore, service restart or release activation has been performed. The scratch directory `/tmp/hermes-memory-synthetic-dlifxuyx` is not the durable ledger for this run; the affected queue/mapping lives in production because of the bug.

## Source fix and tests

- The runner now uses the existing core `scoped_settings` constructor, which rebinds all canonical paths together.
- A separate fail-closed guard verifies that `data_dir`, `db_path` and `blob_dir` are independent descendants of the scratch root and that the evaluation bank is distinct, before opening a writer or backend socket.
- Regression tests reproduce the original partial-replace bug and reject each escaped production path and a production-bank target.
- Future test ledgers are retained for reconciliation rather than deleted automatically on uncertain cleanup. A separate reviewed cleanup is still required if the backend operation cannot be proved terminal.

References: [runner](/home/jugaadu/Projects/hermes-memory/evals/run_memory_intelligence.py), [isolation regressions](/home/jugaadu/Projects/hermes-memory/tests/unit/test_live_eval_isolation.py), [core scoped settings](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/config.py:419), [migration definitions](/home/jugaadu/Projects/hermes-memory/src/hermes_memory/storage/migrations.py).

## Owner direction and continuing repair

The owner subsequently stated: “i don't need old data.. it's all testinf data.. so don't worry about data..keep fixing issues”. This removes the requirement to preserve the existing test corpus for migration compatibility and authorizes continued repairs. It does not retroactively make the isolation failure correct, nor authorize unrelated deletion. No data deletion is necessary for the source fixes recorded in the implementation progress document.

Prefer a backed-up, verified **forward-compatible repair** to a blind schema downgrade. Before any activation, verify the actual installed backend/provider import paths, isolate the schema incompatibility, review a compatible source/release, and use the owned installation workflow. Current source includes incomplete new packages and cannot be declared deployment-ready merely from passing tests.

Separately review removal of exactly the three synthetic records through the existing owned erasure workflow, settlement of the test-related queue/mapping, and cleanup of the exact test bank/operation metadata. Preserve the original personal record and all durable erasure, inquiry and delivery fences. Do not restore an old snapshot or manually delete migration entries as a shortcut.

Owner approval is no longer the blocker. The remaining deployment work is technical: finish and verify coherent runtime/backend/provider staging and activation. No service restart or release activation has yet been performed during this continuation.

Continuation: nine additional defects and regression evidence are recorded in the [repair register](memory-intelligence-repair-register.md). A new schema-17-compatible runtime/backend candidate was staged and independently checked, but it was not activated. The current pointer and running installation are unchanged; live health recovery is still not claimed. Existing test data has not needed deletion for these source and packaging fixes.

## Subsequent runtime repair

The preceding continuation statements are historical. Section 8 of the [implementation progress](memory-intelligence-implementation-progress.md) records successful core activation of `snapshot-a0840de1caad49b5`, owned-service health checks and a new properly isolated retain/recall smoke run. The schema-compatibility failure is no longer blocking that tested flow. No schema downgrade, migration-history edit or data deletion was used. This does not prove cleanup of all metadata from the original incident, full roadmap completion, semantic recall quality or learned reranking. Outbound memory delivery remains paused during validation.
