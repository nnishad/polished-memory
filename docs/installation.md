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

## Configuration

The runtime reads one owned file, `$HERMES_MEMORY_HOME/hermes-memory.env`, then the
process environment (process wins, so a systemd drop-in can override the file).
`deployment/env/hermes-memory.env.example` is the annotated copy; the same for the backend
at `deployment/env/hindsight.env.example`. Keep both at mode `0600`.

Three rules are enforced at load time rather than left to the operator's memory:

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

`HERMES_MEMORY_OWNER_PRINCIPAL` is left unset on purpose. With no owner named an erasure
can be previewed but never confirmed, which fails closed instead of accepting any caller
that claims to be the owner. It has to be a human login; an agent-role credential must
never be able to confirm a forgetting, a revocation, an identity or a budget.

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
hermes-memory retire --profile work --reason "the profile moved"   # unlink
```

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
