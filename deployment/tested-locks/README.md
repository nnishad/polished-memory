# Tested locks

A lock here is not a suggestion of what to install. It is the exact resolved set the
test suite and the synthetic harness were run against, kept so that a later failure can
be asked a useful question: did the code change, or did the world under it?

| File | Resolved for | Verified by |
| --- | --- | --- |
| `uv-py3.12.lock` | CPython 3.12.13, `uv lock` output of this tree | `uv run pytest` (2653 passed, 1 skipped) and `uv run python evals/run_synthetic.py` (56 of 56 measured checks) |
| `../dependencies.json` | the same lock, with each package's declared license | generated offline from resolved metadata; nothing was installed to produce it |

Two things are deliberate and should not be "fixed" by a dependency bot:

- **The runtime needs no packages at all.** `pyproject.toml` lists `dependencies = []`.
  The memory core — capture, the canonical store, blobs, lineage, erasure, snapshots,
  lexical retrieval, the gate, the queue, the operator doors — runs on the standard
  library. `hindsight-client` is an extra used only by the optional bridge, and `pytest`
  is a dev extra. A lock that resolves them is a record of what the tests touched, not a
  licence for the runtime to import them.
- **No model, embedding or cloud package appears anywhere.** Inference happens by HTTP to
  an endpoint this installation names, restricted to approved loopback or LAN hosts. If a
  future lock adds a vendor SDK, that is the change to argue about before it is merged,
  because it silently widens what "inference is local" means.

Regenerate with `uv lock`, then re-run the suite and the harness and record the counts
above. `uv pip install` is deliberately not part of any step: the dev environment is the
operator's.
