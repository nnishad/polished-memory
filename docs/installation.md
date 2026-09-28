# Installing hermes-memory

This describes the installer that exists in this tree, not the one the plan still
promises. Where a step is deliberately absent — nothing drains the archive on its own,
`upgrade` performs no switch, `uninstall` cannot delete memory — that is stated as the
behaviour rather than glossed over.

Everything here is a user-level operation. The installer never uses `sudo`, never touches
system-wide units, and only ever names its own three unit files to `systemctl --user`.

## What has to be true first

Two separate environments have to be unpacked before setup will go through, because
packaging is a release artefact rather than something an installer does at 02:00 on a
Thursday:

| Path | Contents | Checked by |
|---|---|---|
| `$HERMES_MEMORY_RELEASE/bin/hermes-memory` | the framework runtime | the `stage` step |
| `$HERMES_MEMORY_RELEASE/hindsight/bin/hindsight-api` | the pinned backend API | the `stage` step |
| `$HERMES_MEMORY_RELEASE/hindsight/bin/python` | the interpreter the worker unit runs | the `stage` step |
| `hermes_memory` and `hindsight_client` importable | the code both processes need | the `stage` step |

The list of programs is read out of the rendered units, so a template edit that starts a
fourth program is checked without anyone remembering to update a list. Anything missing is
reported as `<path> is not staged; unpack the pinned release there or point
HERMES_MEMORY_RELEASE at it`, and the transaction stops before a byte is written.

`hindsight_client` is only demanded when a backend route is configured. A capture-only
installation is a supported operating state and installs with nothing but the framework.

Alongside those environments the release ships `deployment/compatibility.json`: the pinned
engine version, the operations the capability table supports, the schema version this build
carries, the plugin's host floor and checkpoint API version, and a digest over the framework
sources. It is generated from the code that enforces each of those facts rather than typed in
next to it, which is what lets `doctor` check the release against itself:

```sh
hermes-memory compatibility --write
```

A running installation compares only the compatibility claims, so editing a source file does
not make `doctor` red; the packaging step compares the digests as well —
`hermes-memory compatibility --digests`, which is what a release build has to pass — so a
patched plugin directory or a tree that was never cut as a release is caught.

The write goes to the tree that is running the command: `HERMES_MEMORY_RELEASE` names the
release being packed, and with nothing named the file lands beside this checkout. A live
installation's `runtime/current` is never a target, even though that is exactly where a
*check* looks — a digest of one tree filed beside another would be read back as agreement, and
the release would carry a compatibility claim no code matches.

## Configuration

The runtime reads one owned file, `$HERMES_MEMORY_HOME/hermes-memory.env`, then the
process environment (process wins, so a systemd drop-in can override the file).
`deployment/env/hermes-memory.env.example` is the annotated copy; the same for the backend
at `deployment/env/hindsight.env.example`. Keep both at mode `0600`.

Five rules are enforced at load time rather than left to the operator's memory:

- **Inference is off until it is named.** `HERMES_MEMORY_INFERENCE_ENABLED=false` is the
  default, and there is no ambient provider or credential discovery.
- **A route must be on the allowlist.** `HERMES_MEMORY_ALLOWED_INFERENCE_HOSTS` admits
  loopback and literal RFC1918 addresses only. Link-local — including
`169.254.169.254` — is refused even when listed. A configured route that is not on the
  list is a startup error, not a fallback.
- **Secrets are named, never inlined.** `HERMES_MEMORY_HINDSIGHT_API_KEY_ENV` holds the
  *name* of the variable carrying the key. Route credentials
  (`HERMES_MEMORY_ROUTE_CREDENTIAL_RETAIN=…`) are opaque tokens that say which operation
  and which physical resource a request means; the upstream keys stay at the gate. Each
  enrolled profile reads its own scoped name, so a second profile cannot sign with the
  first one's key — and a profile with no scoped secret gets no route instead of
  borrowing one.
- **Delivery needs a destination.** `HERMES_MEMORY_DELIVERY_ENABLED` without
  `HERMES_MEMORY_DELIVERY_TARGET` refuses to load: an approved address is what makes a
  message to another person the owner's decision rather than a capability the runtime has
  on its own, and a broadcast scheme is refused on the same grounds.
- **An evaluator is a program, not a word on a command line.**
  `HERMES_MEMORY_EVALUATOR_COMMAND` has to be an absolute path, because it names the only
  thing that can turn a run into a rule and a bare word resolves from whichever directory
  `PATH` happens to offer. `HERMES_MEMORY_EVALUATOR_ENV` is checked the same way: entries
  that are not variable names are refused at load rather than silently dropped later.

`HERMES_MEMORY_OWNER_PRINCIPAL` is left unset on purpose. With no owner named an erasure
can be previewed but never confirmed, which fails closed instead of accepting any caller
that claims to be the owner. It has to be a human login; an agent-role credential must
never be able to confirm a forgetting, a revocation, an identity or a budget.

## Nothing comes from the old installation

This is a fresh installation. The file list in the plan names a `docs/migration.md`, and it
is deliberately absent: there is no importer, no upgrade path, and no compatibility read of
the retired framework's databases. Concretely:

- No command reads an old `M/personal_memory` database, its separate deletion ledger, its
  learning objects, or the contents of its banks.
- Nothing is re-keyed. A canonical id here is derived by this build alone, from
  `(source, source_id, revision)`, so a fact or an identity confirmation remembered over
  there resolves to nothing here — and is not silently trusted if a string happens to match.
- Identity starts empty. The owner confirms joins again, from evidence this store holds:
  inheriting a merge decided under somebody else's authority is the exact mistake the
  owner-only confirmation rule exists to prevent.
- The old installations were torn down with the owner's approval rather than left
  half-adopted, which is also what retired the migration code and the door that would have
  needed it.

What is left of that history is an honest absence: an archive that claims nothing about what
it did not ingest, and a memory epoch that starts at one.

## Setup

```sh
hermes-memory setup --hermes-home ~/.hermes/profiles/work \
    --ref <40-character release commit>                      # plan only
hermes-memory setup --hermes-home ~/.hermes/profiles/work \
    --profile work --ref <40-character release commit> \
    --actor <your login> --review <digest the plan printed>
```

Without `--ref` the plan still prints, and `register-plugin` is listed as blocking: the
commit is what makes an installed plugin a reviewed artefact rather than whatever was in the
working tree this morning.

The first form runs the eleven §10.4 steps in plan mode and prints what each would do,
plus the step digests and a single `review_digest`. The second applies it, and is refused
unless the digest still matches the plan that was shown — so an approval does not survive
a change underneath it. Advisory readings (a wall clock, a total the pass itself moves)
are kept out of the digest; `blocking` is in it, because a pause set after an approval
should invalidate that approval.

The steps, in order: `inventory`, `plan`, `stage`, `configure`, `initialize`,
`register-plugin`, `preflight`, `activate`, `services`, `canary`, `finish`. Steps 1–2
read, 3–9 write, 10 proves, 11 reports. `register-plugin` wraps the supported host
commands (`hermes plugins install "file://<this checkout>#integrations/hermes-memory" --ref …`
then `hermes plugins enable hermes-memory --no-allow-tool-override`) and `activate` writes the
provider selection through the host's narrow dotted-key writer, after approval, leaving
`model.*` alone. The whole list of host commands is printed with the plan, so a reviewer sees
every external action before any of them runs.

Re-running setup against a machine that is already serving is the ordinary case rather than a
collision, and `stop` first is not required. The `inventory` step reads the kernel's socket
table and, for a port it wants, names the process holding it by socket inode and by the command
line that process carries. When the holder is one of this installation's own processes the held
port is said as an advisory — *a setup run here replaces that listener* — and the plan is not
blocked; the sentence names the program and PID, read from `/proc`, and no host command is run
to find out. A listener that cannot be attributed to this installation's home keeps blocking the
run, and the report names that as the missing fact instead of guessing at it. Which of its own
processes happened to be up is outside what the digest covers, so bouncing a service does not
expire an approval that was about something else. A backup copy of a store, taken under the
profile's own `snapshots/` directory, is read as one copy of one owner rather than as a second
capture owner.

The same split is what a script is allowed to ask about:

```sh
hermes-memory inventory --conflicts    # prints the collisions, exits 1 if there is one
```

It prints the JSON array of the sentences that would stop a setup run and takes exit status 1
if that array is non-empty and 0 if it is not, so a healthy installation that is up and
serving — whose own listeners are the most interesting fact about it — is reported as not
blocked. Plain `hermes-memory inventory` is the door for everything worth reading before a
run, advisories included, and its exit status never carries an opinion about whether to
proceed: the advice half of the split is for a human, and only the collision half gates.

One of the facts the inventory reads is the host's own registration record. §10.3 pairs the
wheel and the plugin from one revision, so the inventory reads the commit the host wrote down
(`~/.hermes/plugins/.install-metadata.json`) beside the commit the staged release carries
(`<release>/RELEASE.json`), and says so when they differ: *the runtime and the host's copy of
the plugin are two different revisions*. Nothing else here can see that split — every other
number in the report still reads as healthy. It is advice rather than a gate, deliberately:
the transaction's own `register-plugin` step already refuses to rewrite a registered plugin
without the host's `--force`, which is the owner's act, and stopping the whole plan at its
first step would block the very release-staging that closes the gap. A release that records no
commit, or a plugin installed by hand, is not reported as a mismatch: an absence is not a
difference.

`--start` and `hermes-memory start` are the only ways anything gets started, and starting
is a separate decision from installing the units. Autostart is a third decision:

```sh
hermes-memory services --autostart enable     # systemctl --user enable, our units only
hermes-memory services --autostart disable    # back again, in reverse dependency order
```

Enabling is not reversible by a reboot, so no command here does it as a side effect of a
`start`, and `start` never lifts a pause: a unit that boots to find inference or delivery
held comes up holding it.

Nothing in setup ingests private data and nothing in setup calls a model. `canary`
captures a synthetic message into a throwaway scope, reads it back with provenance and
forgets it; the model-backed version is `doctor --synthetic-probe`, which is opt-in by
name.

## Enrolment and profiles

`enroll` maps one Hermes profile home to its own memory — its own store, bank, scopes and
credential scope — and is the same reviewed-plan-then-digest dance. It is what
`hermes memory setup hermes-memory` leaves for the owner to run: the host's wizard writes
this plugin's configuration and selects it as the provider, which is the host's decision to
make, and then ends by handing back the exact `hermes-memory enroll --hermes-home …` command
that finishes the mapping. That is deliberate — a digest computed by the code that will
consume it proves nothing about what a person was shown, while a digest handed back by the
owner, after the paths were printed, does. The framework never calls the host's setup
command from inside its own flow, so there is exactly one orchestration owner and no
recursive wizard.

```sh
hermes-memory profiles                      # what this installation serves
hermes-memory enroll --hermes-home …        # plan
hermes-memory init --hermes-home …          # open that profile's own store
hermes-memory retire --profile work --reason "the profile moved"   # unlink
```

Enrollment is a mapping and writes no archive: `profiles` reports `store_present` false
until somebody opens it. `init --hermes-home <home>` does exactly that — creates and
migrates the store and the blob directory beside it — and refuses a home nobody enrolled,
because creating one from a shell command is how a second memory appears with no owner
decision behind it. `setup` performs the same step as part of the installation
transaction; having `init` for it is what lets a memory exist before, or without, the
plugin being registered with the host.

A door that reads one memory (`status`, `doctor`, `measure`, `sources list`, `audit`,
`explain`) answers from the single enrolled profile when there is one, and refuses to
choose between two.

Retiring a profile unlinks it. It does not forget anything: the plan's rule is that
disabling a source or a profile revokes authorisation for what comes next, and says
nothing about what is already stored.

## Services

Three owned user units, always named in this order, because the dependency is real:

1. `hermes-memory.service` — the runtime: canonical store, context broker, admission
   endpoint. `ExecStart=<release>/bin/hermes-memory serve`.
2. `hermes-memory-hindsight.service` — the backend API on a private loopback port. It is
   the only process that owns the pg0 lifecycle and migrations, and gets its own data
   location instead of implicitly selecting an old `~/.pg0/instances`.
3. `hermes-memory-worker.service` — the background worker, `BindsTo` the API. Its
   `ExecStart` is `python -m hermes_memory.backend.worker_launcher`, not the
   distribution's own script, because §8.1.2 attribution has to be written down in the
   process that dispatches the task.

All three run under `ProtectSystem=strict`, `ProtectHome=read-only`, `NoNewPrivileges`,
`PrivateTmp`, a `UMask=0077` and their own memory and task ceilings, with
`ReadWritePaths` limited to the instance home, the backend data directory and the pg0
directory. The units are rendered from `deployment/systemd/` and refuse to install if a
template names a path outside the installation.

`start` brings the gate up before the backend and the backend before the worker; `stop`
persists the pause *before* it touches a process, so a restart cannot resume formation of
its own accord. The resolved database DSN is passed privately from the API process to the
worker; no command in this tree prints it, and status and doctor print redacted detail.

```sh
hermes-memory services                                  # show the plan
hermes-memory services --install <digest> --actor "$USER"
hermes-memory services --autostart enable
hermes-memory start
hermes-memory stop
```

A unit that already exists and was not written by this installation is reported and left
byte-for-byte alone. Ownership is recorded as a digest of each unit we wrote, in
`services.json` beside them in the user unit directory (mode `0600` in a `0700` directory),
so a file edited by hand after we wrote it is named rather than silently taken over.
`systemctl --user daemon-reload` happens only after one of our own files actually changed.

The units and the compatibility manifest both come from a *release tree* — an unpacked
release with `deployment/` beside the `bin/` it names, pointed at by
`HERMES_MEMORY_RELEASE` or by `<instance home>/runtime/current`. A wheel installed on its
own carries neither, and the doors say so instead of inventing a path inside the venv:
`services`, `start`, `upgrade` and `uninstall` refuse with the release pointer to set, and
`doctor`'s `release` finding is a warning that this build states no compatibility claim.
The reading of that warning is the same one the manifest exists to support: a release that
ships no manifest of its own is not vouched for by whichever checkout happens to be on the
disk.

