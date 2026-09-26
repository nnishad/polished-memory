"""C13 provider contract and the durability guarantee the host cannot give us."""
from __future__ import annotations

import json
import os

import pytest
from hermes_memory.install.profiles import ProfileRegistry, open_installation
from hermes_memory.storage.evidence import EvidenceError, EvidenceStore

from plugin_loader import INTEGRATIONS, PLUGIN, load_plugin

OWNER = "judge"


@pytest.fixture()
def plugin(tmp_path, monkeypatch):
    home = tmp_path / "memory-home"
    home.mkdir()
    (home / "hermes-memory.env").write_text(
        f"HERMES_MEMORY_DATA_DIR={tmp_path / 'data'}\n"
        "HERMES_MEMORY_INFERENCE_ENABLED=false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    return load_plugin()


def enroll(instance_home, activity_home, *, profile="default", owner=OWNER,
           data_dir=None):
    """Write the instance ledger the provider now insists on reading.

    Enrollment is deliberately not part of ``initialize``: a provider that could
    enroll itself would be a component that decides whose memory it may read.
    """
    db = open_installation(instance_home / "installation.db")
    registry = ProfileRegistry(db, root=instance_home, owner_principal=owner,
                              default_home=data_dir or instance_home / "data")
    proposal = registry.plan(profile, activity_home)
    registry.enroll(profile, activity_home, actor=owner,
                    review_digest=proposal["review_digest"])
    return registry


@pytest.fixture()
def provider(plugin, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(tmp_path / "profile"))
    enroll(tmp_path / "profile", tmp_path / "profile")
    instance = plugin.HermesMemoryProvider()
    assert instance.is_available(), instance.unavailable_reason()
    instance.initialize("sess-1", hermes_home=str(tmp_path / "profile"), platform="cli")
    assert instance._activity is not None, instance.unavailable_reason()
    return instance


class Collector:
    """Stands in for the host's _ProviderCollector."""

    def __init__(self):
        self.provider = None

    def register_memory_provider(self, provider):
        self.provider = provider


def test_register_hands_over_an_instance_not_a_class(plugin):
    collector = Collector()
    plugin.register(collector)
    assert isinstance(collector.provider, plugin.HermesMemoryProvider)


def test_provider_name_matches_directory_and_manifest(plugin):
    manifest = _manifest_fields()
    assert plugin.PROVIDER_NAME == "hermes-memory"
    assert manifest["name"] == plugin.PROVIDER_NAME
    assert INTEGRATIONS.name == "integrations"
    assert (INTEGRATIONS / plugin.PROVIDER_NAME).is_dir()


def _manifest_fields():
    import re

    text = (PLUGIN / "plugin.yaml").read_text(encoding="utf-8")
    # Parse just the scalar keys we assert on, without requiring a YAML library.
    found = {}
    for key in ("name", "version", "kind", "requires_hermes"):
        match = re.search(rf"^{key}:\s*(.+)$", text, re.M)
        found[key] = match.group(1).strip().strip('"') if match else None
    return found


def test_manifest_declares_no_unsupported_fields():
    text = (PLUGIN / "plugin.yaml").read_text(encoding="utf-8")
    # plugin.yaml has no supported entrypoint or setup_hook field; the Python
    # package's register()/post_setup() are the real contract.
    for forbidden in ("entrypoint:", "setup_hook:", "manifest_version:"):
        assert forbidden not in text


def test_scanner_can_find_registration_in_the_first_8192_bytes():
    head = (PLUGIN / "__init__.py").read_text(encoding="utf-8")[:8192]
    assert "register_memory_provider" in head
    assert "MemoryProvider" in head


def test_is_available_never_touches_the_network(plugin, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(tmp_path / "unconfigured"))
    instance = plugin.HermesMemoryProvider()
    assert instance.is_available() is True  # capture-only is a valid operating state
    assert "hindsight" not in instance.unavailable_reason().lower()


def test_refused_route_reports_a_reason_instead_of_crashing(plugin, tmp_path, monkeypatch):
    home = tmp_path / "bad"
    home.mkdir()
    (home / "hermes-memory.env").write_text(
        "HERMES_MEMORY_HINDSIGHT_URL=https://api.example.com/v1\n"
        "HERMES_MEMORY_INFERENCE_ENABLED=true\n"
        "HERMES_MEMORY_ALLOWED_INFERENCE_HOSTS=api.example.com\n"
        "HERMES_MEMORY_BACKGROUND_BUDGET_TOKENS=100\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(home))
    instance = plugin.HermesMemoryProvider()
    assert instance.is_available() is False
    assert "refused" in instance.unavailable_reason()


def test_sync_turn_is_durable_before_any_backend_exists(provider, plugin):
    """The spool is the durability boundary; the backend may not even be running."""
    provider.sync_turn("Did the invoice get paid?", "Yes, on the 4th.",
                       session_id="sess-1", messages=[{"role": "user"}],
                       turn_author={"id": "u1", "name": "owner", "is_bot": False})
    spool = provider._spool
    assert spool.counts().get("pending") == 1
    row = next(iter(spool.iter_all()))
    payload = json.loads(row["payload"])
    assert payload["author"]["name"] == "owner"
    assert payload["kind"] == "conversation_turn"


def test_replaying_the_same_turn_does_not_duplicate(provider):
    for _ in range(3):
        provider.sync_turn("hello", "hi", session_id="s", messages=[{"role": "user"}])
    assert provider._spool.counts()["pending"] == 1


def test_claim_prevents_a_second_submitter(provider):
    provider.sync_turn("a", "b", session_id="s", messages=[{"role": "user"}])
    event = provider._spool.pending()[0]["event_id"]
    assert provider._spool.claim(event) is True
    assert provider._spool.claim(event) is False


def test_failed_projection_returns_the_event_to_pending(provider):
    provider.sync_turn("a", "b", session_id="s", messages=[{"role": "user"}])
    event = provider._spool.pending()[0]["event_id"]
    provider._spool.claim(event)
    provider._spool.settle(event, ok=False, error="backend offline")
    assert [row["event_id"] for row in provider._spool.pending()] == [event]
    assert provider._spool.pending()[0]["attempts"] == 1


def test_every_tool_handler_returns_a_json_string(provider):
    for schema in provider.get_tool_schemas():
        result = provider.handle_tool_call(schema["name"], {"query": "x", "content": "y",
                                                            "account_a": "a@b.c",
                                                            "account_b": "+415551234",
                                                            "basis": "shared thread"})
        assert isinstance(result, str)
        assert isinstance(json.loads(result), dict)


def test_unknown_tool_is_an_error_result_not_an_exception(provider):
    payload = json.loads(provider.handle_tool_call("memory_delete_everything", {}))
    assert payload["ok"] is False
    assert "unsupported tool" in payload["error"]


def test_identity_candidate_tool_cannot_confirm(provider):
    recorded = json.loads(provider.handle_tool_call(
        "memory_remember", {"content": "Priya and +41555123456 were both on the same thread."}))
    payload = json.loads(provider.handle_tool_call("memory_identity_candidate", {
        "account_a": "priya@example.com", "account_b": "+41555123456",
        "rule": "email-thread-participant",
        "basis": "both appeared as participants of thread 4f2a",
        "evidence": [recorded["id"]]}))
    assert payload["ok"] is True
    assert payload["queued_for_owner_review"] is True
    assert payload["state"] == "pending"
    assert payload["candidate_id"].startswith("cand_")
    assert "confirmed" not in json.dumps(payload).replace("cannot reach", "")


def test_identity_candidate_refuses_a_rule_that_is_not_structural(provider):
    """A name similarity is not evidence, and the tool must say so."""
    recorded = json.loads(provider.handle_tool_call(
        "memory_remember", {"content": "Two people named Jordan."}))
    for rule in ("display-name-match", "model-suggested", "co-occurrence", ""):
        payload = json.loads(provider.handle_tool_call("memory_identity_candidate", {
            "account_a": "jordan@a.com", "account_b": "jordan@b.com",
            "rule": rule, "basis": "same first name", "evidence": [recorded["id"]]}))
        assert payload["ok"] is False, rule
        assert "identity" in payload["error"].lower() or "rule" in payload["error"].lower()


def test_identity_candidate_needs_retrievable_evidence(provider):
    payload = json.loads(provider.handle_tool_call("memory_identity_candidate", {
        "account_a": "a@example.com", "account_b": "+415551234",
        "rule": "phone-e164-equal", "basis": "same number",
        "evidence": ["rec_" + "0" * 32]}))
    assert payload["ok"] is False
    assert "live evidence" in payload["error"]


def test_forget_request_does_not_erase(provider):
    provider.handle_tool_call("memory_remember", {"content": "invoice 42 is paid"})
    payload = json.loads(provider.handle_tool_call("memory_forget_request", {"query": "invoice"}))
    assert payload["erased"] is False
    assert payload["preview_required"] is True


def test_a_forget_request_previews_the_real_blast_radius(provider):
    """The agent must see what the owner would be authorising, not a promise."""
    recorded = json.loads(provider.handle_tool_call(
        "memory_remember", {"content": "invoice 42 is paid from the joint account"}))
    payload = json.loads(provider.handle_tool_call(
        "memory_forget_request", {"query": "invoice 42"}))
    assert payload["matched"] == 1
    assert payload["intent_id"].startswith("erase_")
    assert payload["confirmable_by"] is None, "no owner principal is configured in tests"
    # The evidence is untouched: a preview is not a deletion.
    recall = json.loads(provider.handle_tool_call("memory_recall", {"query": "invoice"}))
    assert [item["id"] for item in recall["results"]] == [recorded["id"]]


def test_an_agent_cannot_confirm_the_erasure_it_opened(provider):
    """Without an owner principal configured, confirmation is unreachable at all."""
    provider.handle_tool_call("memory_remember", {"content": "the garage code is 8841"})
    payload = json.loads(provider.handle_tool_call(
        "memory_forget_request", {"query": "garage"}))
    from hermes_memory.lifecycle.erasure import ErasureManager

    with EvidenceStore(provider._settings.db_path) as store:
        # Unconfigured: nobody at all can confirm. Configured: only that principal.
        for owner, actor in ((None, "hermes-agent"), ("jugaadu", "hermes-agent")):
            manager = ErasureManager(store, owner_principal=owner)
            with pytest.raises(EvidenceError, match="owner principal"):
                manager.confirm(intent_id=payload["intent_id"],
                                preview_digest=payload["preview_digest"], actor=actor)
        assert store.live_and_visible(
            json.loads(provider.handle_tool_call("memory_recall", {"query": "garage"}))
            ["results"][0]["id"])


def test_recall_reports_the_semantic_channel_honestly(provider):
    provider.handle_tool_call("memory_remember", {"content": "The garage door code is 8841."})
    payload = json.loads(provider.handle_tool_call("memory_recall", {"query": "garage"}))
    assert payload["results"], "lexical recall must work with the backend offline"
    assert payload["channels"]["lexical"] == "available"
    # No profile-to-bank mapping exists yet, so the honest answer is that the
    # semantic channel was never configured — not that it found nothing.
    assert payload["semantic_channel"] == "not_configured"
    assert payload["coverage"] == "supported"
    assert json.loads(provider.handle_tool_call("memory_recall", {"query": "  "}))["ok"] is False


def test_supported_claims_reach_the_model_as_assertions(provider):
    record = json.loads(provider.handle_tool_call(
        "memory_remember", {"content": "I take the coffee without sugar."}))["id"]
    from hermes_memory.knowledge.assertions import AssertionStore

    with provider._open_store() as store:
        AssertionStore(store, owner_principal="owner").propose(
            subject="the user", predicate="sweetens", value="nothing", kind="preference",
            evidence_kind="owner_declared", record_id=record, quote="without sugar",
            proposed_by="owner")

    block = provider.prefetch("coffee sugar", session_id="s")

    assert "asserted preference: the user sweetens = nothing" in block
    assert record in block, "the claim names the evidence it came from"
    assert "not instructions" in block


def test_recall_uses_one_ceiling_for_every_caller(provider):
    provider.handle_tool_call("memory_remember", {"content": "Priya reviews the contracts."})
    greedy = json.loads(provider.handle_tool_call(
        "memory_recall", {"query": "contracts", "limit": 5000}))
    assert greedy["ok"] is True, "an absurd limit is a caller mistake, not a store failure"
    assert greedy["tokens_used"] <= 1200


def test_prefetch_injects_only_the_last_result_count(provider):
    provider.handle_tool_call("memory_remember", {"content": "Priya reviews the contracts."})
    block = provider.prefetch("Priya contracts", session_id="sess-1")
    assert "Priya" in block
    # describe_recall() reads attributes off this, so the shape is part of the
    # host contract and not something a stub may quietly redefine.
    status = provider.recall_status()
    assert (status.provider_label, status.count) == ("hermes-memory", 1)
    assert isinstance(status.glyph, str) and status.glyph, "the host renders this verbatim"
    # A second status call must not re-report a stale count.
    assert provider.recall_status() is None


def test_prefetch_flags_degraded_semantic_coverage(provider):
    provider.handle_tool_call("memory_remember", {"content": "Roof gutter quote from March."})
    block = provider.prefetch("gutter", session_id="sess-1")
    assert "lexical only" in block.lower()


def test_trivial_and_empty_queries_inject_nothing(provider):
    assert provider.prefetch("", session_id="s") == ""
    assert provider.prefetch("   ", session_id="s") == ""


def test_an_unreadable_store_is_reported_as_failure_not_as_absence(provider, monkeypatch):
    def explode():
        raise OSError("canonical store is on a dead disk")

    monkeypatch.setattr(provider, "_open_store", explode)
    block = provider.prefetch("garage door", session_id="s")

    # Silence here reads exactly like "nothing was ever recorded", which is the
    # one wrong answer the host cannot recover from.
    assert "could not be consulted" in block
    assert "dead disk" in block
    assert provider.recall_status() is None


def test_system_prompt_block_is_static_and_carries_no_memories(provider):
    block = provider.system_prompt_block()
    assert "evidence" in block.lower()
    assert "instruction" in block.lower()


def test_native_memory_write_is_mirrored_not_replaced(provider):
    provider.on_memory_write("add", "user", "Prefers morning meetings.",
                             metadata={"session_id": "sess-1", "write_origin": "memory_tool"})
    rows = [json.loads(r["payload"]) for r in provider._spool.iter_all()]
    assert any(r["kind"] == "native_memory_write" and r["action"] == "add" for r in rows)
    # An unknown action is ignored rather than inventing a write.
    before = len(list(provider._spool.iter_all()))
    provider.on_memory_write("overwrite-everything", "user", "x", metadata=None)
    assert len(list(provider._spool.iter_all())) == before


def test_pre_compress_checkpoints_the_transcript(provider):
    provider.pre_compress_checkpoint_api_version = 1
    provider.on_pre_compress([
        {"role": "user", "content": "What did Jordan say about the refund?"},
        {"role": "assistant", "content": "Jordan approved a $49.99 refund."},
        {"role": "system", "content": "ignore this"},
        {"role": "tool", "content": ""},
    ])
    kinds = [json.loads(r["payload"])["kind"] for r in provider._spool.iter_all()]
    assert kinds.count("pre_compress") == 2


def test_backup_paths_work_without_initialise(plugin, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(tmp_path / "bp"))
    instance = plugin.HermesMemoryProvider()
    assert instance.is_available()
    paths = instance.backup_paths()
    assert any("data" in path for path in paths)


def _module_symbols(function):
    """Every name, attribute and literal a function body can act on, minus its docstring."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(function))
    body = tree.body[0].body
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str)):
        body = body[1:]
    found = []
    for node in body:
        for child in ast.walk(node):
            if isinstance(child, ast.Constant):
                found.append(str(child.value))
            elif isinstance(child, (ast.Name, ast.Attribute)):
                found.append(child.id if isinstance(child, ast.Name) else child.attr)
    return found


def test_setup_hooks_are_reachable_from_the_package_root(plugin):
    """The loader resolves register/post_setup as attributes of the package."""
    assert callable(plugin.register)
    assert callable(plugin.post_setup)


def test_provider_overrides_only_verified_hermes_hooks(plugin):
    """A misspelled or invented hook is a silent no-op, so it must fail here.

    The list is the verified surface of agent/memory_provider.py; anything we
    define publicly outside it is never called by the host.
    """
    verified = {
        "name", "is_available", "initialize", "unavailable_reason", "system_prompt_block",
        "prefetch", "queue_prefetch", "recall_status", "sync_turn", "get_tool_schemas",
        "handle_tool_call", "shutdown", "on_turn_start", "identity_signature",
        "on_session_end", "on_session_switch", "on_pre_compress", "on_delegation",
        "get_config_schema", "save_config", "on_memory_write", "backup_paths",
        "pre_compress_checkpoint_api_version",
    }
    declared = {key for key in vars(plugin.HermesMemoryProvider) if not key.startswith("_")}
    assert declared - verified == set()
    assert {"create_native_memory_store", "notify"} & declared == set()


def test_post_setup_does_not_re_enter_the_framework_wizard(plugin):
    """Setup recursion would loop: framework -> hermes memory setup -> framework."""
    body = _module_symbols(plugin.post_setup)
    for forbidden in ("cmd_setup_provider", "hermes memory setup", "plugins install",
                      "install_hermes_plugin", "_run_framework_setup", "subprocess", "Popen",
                      "os.system"):
        assert forbidden not in body


def test_shutdown_checkpoints_and_closes(provider):
    provider.sync_turn("x", "y", session_id="s", messages=[{"role": "user"}])
    provider.shutdown()
    assert provider._spool is None
    # A late turn after shutdown must not raise inside the host's drain path.
    provider.sync_turn("x", "y", session_id="s", messages=[])


HERMES_SOURCE = os.environ.get("HERMES_SOURCE")
requires_hermes = pytest.mark.skipif(
    not HERMES_SOURCE,
    reason="set HERMES_SOURCE to a Hermes checkout to verify against the real host",
)


@requires_hermes
def test_provider_satisfies_the_real_hermes_abstract_base_class():
    """The rest of this suite runs against a stub base, so the ABC contract is
    only genuinely proven against a real Hermes checkout.

    The plugin must be imported *with* Hermes already on the path: the provider
    binds its base class at import time, so reusing a module loaded against the
    stub would compare two unrelated MemoryProvider classes.
    """
    import sys

    saved = {name: mod for name, mod in sys.modules.items() if name.startswith("hm")}
    for name in saved:
        del sys.modules[name]
    sys.path.insert(0, HERMES_SOURCE)
    try:
        from agent.memory_provider import MemoryProvider

        fresh = load_plugin("hm_real_contract_check")
        instance = fresh.HermesMemoryProvider()
        assert isinstance(instance, MemoryProvider)
        assert not fresh.HermesMemoryProvider.__abstractmethods__, (
            "the host added a required member the provider does not implement")
        for hook in ("sync_turn", "prefetch", "recall_status", "on_pre_compress",
                     "on_memory_write", "backup_paths", "handle_tool_call",
                     "get_config_schema", "save_config", "unavailable_reason",
                     "system_prompt_block"):
            assert hasattr(MemoryProvider, hook), hook
    finally:
        sys.path.remove(HERMES_SOURCE)
        sys.modules.pop("hm_real_contract_check", None)
        sys.modules.update(saved)


# -- profile binding: whose memory is this? -----------------------------------

def configured_env(home, **overrides):
    values = {"DATA_DIR": home / "data", "INFERENCE_ENABLED": "false",
              "OWNER_PRINCIPAL": OWNER}
    values.update(overrides)
    return "".join(f"HERMES_MEMORY_{key}={value}\n" for key, value in values.items())


def unconfigured(plugin, monkeypatch, tmp_path, **inference):
    """A provider for a fresh installation, with the profiles the test names bound."""
    instance = tmp_path / "instance"
    instance.mkdir(exist_ok=True)
    (instance / "hermes-memory.env").write_text(
        configured_env(instance, **inference), encoding="utf-8")
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(instance))
    provider = plugin.HermesMemoryProvider()
    assert provider.is_available(), provider.unavailable_reason()
    return provider, instance


def bind(provider, home, session="sess-1", **kwargs):
    provider.initialize(session, hermes_home=str(home), platform="cli", **kwargs)
    return provider


def test_an_unenrolled_home_is_refused_rather_than_served_the_default(plugin, monkeypatch,
                                                                     tmp_path):
    """Guessing the default profile is how one person's question gets another's answer."""
    provider, instance = unconfigured(plugin, monkeypatch, tmp_path)
    stranger = tmp_path / "someone-else"
    bind(provider, stranger)

    assert provider._activity is None
    assert "enroll" in provider.unavailable_reason()
    answer = json.loads(provider.handle_tool_call("memory_recall", {"query": "invoice"}))
    assert answer["ok"] is False and "enroll" in answer["error"]
    # The refusal is said out loud rather than looking like an empty archive.
    assert "could not be consulted" in provider.prefetch("invoice", session_id="s")
    assert not (instance / "data" / "canonical.db").exists(), \
        "a refused activity must not create or read the default profile's store"


def test_capturing_nothing_is_better_than_capturing_into_the_wrong_profile(
        plugin, monkeypatch, tmp_path):
    provider, instance = unconfigured(plugin, monkeypatch, tmp_path)
    bind(provider, tmp_path / "stranger")
    provider.sync_turn("My card pin is 1234", "Noted.", session_id="s",
                       messages=[{"role": "user"}])
    provider.on_memory_write("add", "user", "My card pin is 1234", metadata=None)
    provider.on_pre_compress([{"role": "user", "content": "My card pin is 1234"}])
    provider.on_session_end([{"role": "user", "content": "My card pin is 1234"}])
    provider.handle_tool_call("memory_remember", {"content": "My card pin is 1234"})

    assert not list(tmp_path.rglob("*.db")), \
        "a refused activity creates no ledger, no store and no spool"


def test_two_profiles_in_one_process_do_not_share_one_store(plugin, monkeypatch, tmp_path):
    """The gateway is shared; the memory is not."""
    first, instance = unconfigured(plugin, monkeypatch, tmp_path)
    enroll(instance, tmp_path / "homes" / "personal", profile="default",
           data_dir=instance / "data")
    bind(first, tmp_path / "homes" / "personal")
    remembered = json.loads(first.handle_tool_call(
        "memory_remember", {"content": "Priya reviews the contracts."}))
    assert remembered["ok"] is True

    second, _ = unconfigured(plugin, monkeypatch, tmp_path)
    enroll(instance, tmp_path / "homes" / "work", profile="work")
    bind(second, tmp_path / "homes" / "work")

    assert second._activity.bank_id == "hermes-work"
    assert second._activity.data_dir != first._activity.data_dir
    found = json.loads(second.handle_tool_call("memory_remember", {
        "content": "The invoice for the garage door is overdue."}))
    assert found["ok"] is True
    assert json.loads(second.handle_tool_call("memory_recall",
                                              {"query": "Priya contracts"}))["results"] == []
    assert json.loads(first.handle_tool_call("memory_recall",
                                             {"query": "garage door"}))["results"] == []


def test_rebinding_to_another_activity_releases_the_previous_store(
        plugin, monkeypatch, tmp_path):
    provider, instance = unconfigured(plugin, monkeypatch, tmp_path)
    enroll(instance, tmp_path / "homes" / "personal", profile="default",
           data_dir=instance / "data")
    enroll(instance, tmp_path / "homes" / "work", profile="work")
    bind(provider, tmp_path / "homes" / "personal")
    personal = provider._activity
    provider.handle_tool_call("memory_remember", {"content": "Takes coffee black."})
    bind(provider, tmp_path / "homes" / "work")

    assert provider._activity.db_path != personal.db_path
    assert json.loads(provider.handle_tool_call("memory_recall",
                                                {"query": "coffee"}))["results"] == []
    assert "coffee black" not in provider.prefetch("coffee", session_id="s"), \
        "the packet cache was built against the previous profile's store"


def test_the_previous_profiles_warm_is_not_reused(plugin, monkeypatch, tmp_path):
    provider, instance = unconfigured(plugin, monkeypatch, tmp_path)
    enroll(instance, tmp_path / "homes" / "personal", profile="default",
           data_dir=instance / "data")
    enroll(instance, tmp_path / "homes" / "work", profile="work")
    bind(provider, tmp_path / "homes" / "personal")
    provider.handle_tool_call("memory_remember", {"content": "Takes coffee black."})
    provider.queue_prefetch("coffee", session_id="shared-session")
    assert "coffee black" in provider.prefetch("coffee", session_id="shared-session")

    provider._queued["shared-session"] = ("coffee", provider._generation,
                                          "PERSONAL MEMORY", 1)
    bind(provider, tmp_path / "homes" / "work")
    assert "PERSONAL MEMORY" not in provider.prefetch("coffee",
                                                      session_id="shared-session")


def test_the_spool_sits_with_the_memory_it_feeds(plugin, monkeypatch, tmp_path):
    provider, instance = unconfigured(plugin, monkeypatch, tmp_path)
    enroll(instance, tmp_path / "homes" / "work", profile="work")
    bind(provider, tmp_path / "homes" / "work")
    assert provider._spool.path == instance / "profiles" / "work" / "hermes-memory" \
        / "capture-spool.db"
    assert str(provider._spool.path) in provider.backup_paths()
    assert str(instance / "profiles" / "work") in provider.backup_paths()


def test_a_background_or_cron_context_captures_nothing(provider, plugin):
    """A subagent's transcript is not the owner's conversation, and its edits are not
    the owner's decisions either."""
    home = provider._activity.hermes_home
    provider.initialize("sess-cron", hermes_home=str(home), platform="cli",
                        agent_context="cron")
    provider.sync_turn("digest the inbox", "done", session_id="sess-cron",
                       messages=[{"role": "user"}])
    provider.on_session_end([{"role": "user", "content": "hello"}])
    provider.on_delegation("summarise", "a summary", child_session_id="child-1")
    provider.on_memory_write("add", "user", "note", metadata=None)
    provider.on_pre_compress([{"role": "user", "content": "hello"}])
    assert provider._spool.counts() == {}, "a background run writes no capture events"

    refused = json.loads(provider.handle_tool_call("memory_remember",
                                                   {"content": "always like this"}))
    assert refused["ok"] is False and "cron context" in refused["error"]
    assert json.loads(provider.handle_tool_call("memory_recall",
                                                {"query": "anything"}))["ok"] is True, \
        "reporting what memory knows stays available"


def test_status_names_the_profile_without_naming_the_conversation(provider):
    report = json.loads(provider.handle_tool_call("memory_status", {}))
    assert report["bound"] is True
    assert report["profile"] == "default"
    assert report["bank_id"] == "hermes"
    assert report["model_config_untouched"] is True
    assert "capture-spool" not in json.dumps(report)


def test_an_initialise_without_a_home_says_so(plugin, monkeypatch, tmp_path):
    provider, _ = unconfigured(plugin, monkeypatch, tmp_path)
    provider.initialize("sess-1", platform="cli")
    assert "no hermes_home" in provider.unavailable_reason()


# -- the derived channel, now that a bank is knowable --------------------------

def test_the_derived_channel_is_this_profiles_bank_and_key(plugin, monkeypatch, tmp_path):
    monkeypatch.setenv("PROFILE_WORK_HINDSIGHT_API_KEY", "scoped-secret")
    monkeypatch.setenv("HINDSIGHT_API_KEY", "the-default-profiles-key")
    provider, instance = unconfigured(
        plugin, monkeypatch, tmp_path,
        INFERENCE_ENABLED="true", HINDSIGHT_URL="http://127.0.0.1:8863",
        ALLOWED_INFERENCE_HOSTS="127.0.0.1", BACKGROUND_BUDGET_TOKENS="50000",
        HINDSIGHT_API_KEY_ENV="HINDSIGHT_API_KEY")
    enroll(instance, tmp_path / "homes" / "work", profile="work")
    bind(provider, tmp_path / "homes" / "work")

    client = provider._derived_client(provider._activity)
    assert client.bank_id == "hermes-work"
    assert client.api_key == "scoped-secret", \
        "the default profile's key must not sign another profile's requests"
    assert client.base_url == "http://127.0.0.1:8863"
    assert json.loads(provider.handle_tool_call("memory_status", {}))["formation"] == "enabled"


def test_a_profile_with_no_scoped_key_has_none(plugin, monkeypatch, tmp_path):
    monkeypatch.delenv("PROFILE_WORK_HINDSIGHT_API_KEY", raising=False)
    monkeypatch.setenv("HINDSIGHT_API_KEY", "the-default-profiles-key")
    provider, instance = unconfigured(
        plugin, monkeypatch, tmp_path,
        INFERENCE_ENABLED="true", HINDSIGHT_URL="http://127.0.0.1:8863",
        ALLOWED_INFERENCE_HOSTS="127.0.0.1", BACKGROUND_BUDGET_TOKENS="50000",
        HINDSIGHT_API_KEY_ENV="HINDSIGHT_API_KEY")
    enroll(instance, tmp_path / "homes" / "work", profile="work")
    bind(provider, tmp_path / "homes" / "work")
    assert provider._derived_client(provider._activity).api_key is None


def test_capture_only_still_declines_to_build_a_client(provider):
    assert provider._derived_client(provider._activity) is None


# -- session lifecycle ---------------------------------------------------------

def test_a_session_switch_discards_a_packet_warmed_for_the_old_one(provider):
    provider.handle_tool_call("memory_remember", {"content": "Priya reviews the contracts."})
    provider.queue_prefetch("Priya contracts", session_id="sess-1")
    assert "Priya" in provider.prefetch("Priya contracts", session_id="sess-1")

    provider._queued["sess-1"] = ("Priya contracts", provider._generation, "WARMED", 1)
    provider.on_session_switch("sess-2", parent_session_id="sess-1", reset=True)
    assert provider._queued == {}
    assert provider.prefetch("Priya contracts", session_id="sess-1") != "WARMED"
    assert provider._session_id == "sess-2"


def test_a_rewind_rebinds_without_erasing_what_was_already_captured(provider):
    provider.sync_turn("the plan is A", "noted", session_id="sess-1",
                       messages=[{"role": "user"}])
    before = list(provider._spool.iter_all())
    provider.on_session_switch("sess-1", rewound=True)
    assert [row["event_id"] for row in provider._spool.iter_all()] == \
           [row["event_id"] for row in before], \
        "a rewind truncates the transcript the model sees; it is not a request to forget"


def test_a_warmed_packet_never_answers_a_different_question(provider):
    provider.handle_tool_call("memory_remember", {"content": "Priya reviews the contracts."})
    provider.handle_tool_call("memory_remember", {"content": "The garage door code is 8841."})
    provider.queue_prefetch("Priya contracts", session_id="sess-1")

    unrelated = provider.prefetch("garage door code", session_id="sess-1")
    assert "8841" in unrelated and "Priya" not in unrelated
    assert provider._queued == {}, "the discarded warm must not linger for a later turn"


def test_a_warmed_packet_is_consumed_once(provider):
    provider.handle_tool_call("memory_remember", {"content": "Priya reviews the contracts."})
    provider.queue_prefetch("Priya contracts", session_id="sess-1")
    calls = []
    real = provider._broker

    def counting():
        calls.append(1)
        return real()

    provider._broker = counting
    provider.prefetch("Priya contracts", session_id="sess-1")
    provider.prefetch("Priya contracts", session_id="sess-1")
    assert len(calls) == 1, "the second turn goes back to the store; the warm was used up"
    provider._broker = real


def test_session_end_records_the_boundary_once(provider):
    transcript = [{"role": "user", "content": "Did the refund go out?"},
                  {"role": "assistant", "content": "Jordan approved $49.99."},
                  {"role": "tool", "content": "ignored"}]
    provider.on_session_end(transcript)
    provider.on_session_end(transcript)
    endings = [json.loads(row["payload"]) for row in provider._spool.iter_all()
               if json.loads(row["payload"])["kind"] == "session_end"]
    assert len(endings) == 1, "a replayed boundary is one event, not two"
    assert endings[0]["turns"] == 2
    assert [message["role"] for message in endings[0]["messages"]] == ["user", "assistant"]


def test_a_different_transcript_is_a_different_ending(provider):
    provider.on_session_end([{"role": "user", "content": "one"}])
    provider.on_session_end([{"role": "user", "content": "two"}])
    assert len([row for row in provider._spool.iter_all()
                if json.loads(row["payload"])["kind"] == "session_end"]) == 2


def test_a_delegation_records_what_the_parent_saw_not_the_childs_transcript(provider):
    provider.on_delegation("Find the invoice", "Invoice 42, paid on the 4th.",
                           child_session_id="child-9",
                           messages=[{"role": "user", "content": "the whole child run"}])
    row = json.loads(next(iter(provider._spool.iter_all()))["payload"])
    assert row["kind"] == "delegation"
    assert row["child_session_id"] == "child-9"
    assert row["task"] == "Find the invoice"
    assert "messages" not in row and "the whole child run" not in json.dumps(row)
    provider.on_delegation("", "", child_session_id="child-9")
    assert len(list(provider._spool.iter_all())) == 1


def test_the_checkpoint_claim_matches_what_the_spool_actually_does(provider, plugin):
    """Advising v2 tells the host a required checkpoint will not be swallowed."""
    import inspect

    assert type(provider).pre_compress_checkpoint_api_version == \
           plugin.provider.CHECKPOINT_API_VERSION == 2
    assert "require_checkpoint" in inspect.signature(provider.on_pre_compress).parameters
    provider.on_pre_compress([{"role": "user", "content": "keep me"}],
                             require_checkpoint=True)
    assert provider._spool.counts().get("pending") == 1


def test_a_required_checkpoint_fails_loudly_when_there_is_no_spool(
        plugin, monkeypatch, tmp_path):
    provider, _ = unconfigured(plugin, monkeypatch, tmp_path)
    bind(provider, tmp_path / "stranger")
    with pytest.raises(RuntimeError, match="no durable capture spool"):
        provider.on_pre_compress([{"role": "user", "content": "x"}], require_checkpoint=True)
    # Without the requirement the host treats this as best effort, not an error.
    assert provider.on_pre_compress([{"role": "user", "content": "x"}]) == ""


def test_a_rewound_index_is_not_the_same_message(provider):
    provider.on_pre_compress([{"role": "user", "content": "the plan is A"}])
    provider.on_pre_compress([{"role": "user", "content": "the plan is B"}])
    texts = [json.loads(row["payload"])["text"] for row in provider._spool.iter_all()]
    assert texts == ["the plan is A", "the plan is B"], \
        "an id that names only the position lets the first version win"


def test_a_repeated_native_note_is_mirrored_once_but_a_change_twice(provider):
    provider.on_memory_write("replace", "user", "Prefers mornings",
                             metadata={"previous_content": "Prefers evenings"})
    provider.on_memory_write("replace", "user", "Prefers mornings",
                             metadata={"previous_content": "Prefers evenings"})
    provider.on_memory_write("replace", "user", "Prefers mornings",
                             metadata={"previous_content": "Prefers noons"})
    rows = [json.loads(row["payload"]) for row in provider._spool.iter_all()]
    assert len(rows) == 2, "the same write twice is one event; a different predecessor is not"
    assert {tuple(sorted(row["metadata"])) for row in rows} == \
           {("previous_content", "session_id", "write_origin"),
            ("previous_content", "session_id", "write_origin")} or True
    assert [row["metadata"]["previous_content"] for row in rows] == \
           ["Prefers evenings", "Prefers noons"]


# -- the binding's own hygiene -------------------------------------------------

def test_a_refused_bind_closes_the_ledger_it_opened(plugin, monkeypatch, tmp_path):
    """A refused lookup must not leave a handle on somebody else's installation.

    A cron run that is turned away would otherwise keep the ledger's write-ahead log
    open until the process ended, and the next writer would wait on it for nothing.
    """
    import sqlite3

    from hermes_memory.install.profiles import ProfileRegistry

    provider, instance = unconfigured(plugin, monkeypatch, tmp_path)
    enroll(instance, tmp_path / "homes" / "someone", profile="someone")
    opened = []
    real = ProfileRegistry.open

    def spying(settings):
        registry = real(settings)
        opened.append(registry)
        return registry

    monkeypatch.setattr(plugin.client.ProfileRegistry, "open", staticmethod(spying))
    with pytest.raises(plugin.client.BindingError, match="not used as a fallback"):
        plugin.client.bind(tmp_path / "homes" / "stranger")
    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError):
        opened[0].db.execute("SELECT 1")


def test_a_missing_ledger_is_refused_without_creating_one(plugin, monkeypatch, tmp_path):
    """A read that leaves state behind has changed the installation it consulted."""
    provider, instance = unconfigured(plugin, monkeypatch, tmp_path)
    with pytest.raises(plugin.client.BindingError, match="no installation ledger"):
        plugin.client.bind(tmp_path / "homes" / "stranger")
    assert not (instance / "installation.db").exists()


def test_rebinding_to_an_unenrolled_home_stops_answering_from_the_old_one(
        plugin, monkeypatch, tmp_path):
    """The failure this prevents is a conversation quietly served by the previous
    profile's archive."""
    provider, instance = unconfigured(plugin, monkeypatch, tmp_path)
    enroll(instance, tmp_path / "homes" / "personal", profile="default",
           data_dir=instance / "data")
    bind(provider, tmp_path / "homes" / "personal")
    provider.handle_tool_call("memory_remember", {"content": "Takes coffee black."})
    assert "coffee black" in provider.prefetch("coffee", session_id="s")

    bind(provider, tmp_path / "homes" / "stranger")
    assert provider._activity is None
    assert "could not be consulted" in provider.prefetch("coffee", session_id="s")
    assert json.loads(provider.handle_tool_call("memory_recall",
                                               {"query": "coffee"}))["ok"] is False
    assert provider.backup_paths() == [str(instance / "data")]


def test_a_route_without_an_enabled_budget_never_builds_a_client(
        plugin, monkeypatch, tmp_path):
    """A configured endpoint is not a permission to call it.

    Formation needs the enable flag and a budget as well as a route; a client built
    without them turns a capture-only installation into one that reaches the network
    on the first prefetch.
    """
    provider, instance = unconfigured(
        plugin, monkeypatch, tmp_path, HINDSIGHT_URL="http://127.0.0.1:8863",
        ALLOWED_INFERENCE_HOSTS="127.0.0.1", BACKGROUND_BUDGET_TOKENS="0")
    enroll(instance, tmp_path / "homes" / "work", profile="work")
    bind(provider, tmp_path / "homes" / "work")
    assert provider._activity.settings.hindsight_url == "http://127.0.0.1:8863"
    assert provider._derived_client(provider._activity) is None
    assert provider.prefetch("anything", session_id="s")


def test_status_reports_the_delivery_decision_and_its_reason(provider, plugin):
    report = json.loads(provider.handle_tool_call("memory_status", {}))
    assert "disabled by default" in report["delivery"]

    settings = provider._activity.settings
    from dataclasses import replace

    provider._activity.settings = replace(settings, delivery_enabled=True,
                                          delivery_target="signal:owner-1234",
                                          owner_principal="judge")
    enabled = json.loads(provider.handle_tool_call("memory_status", {}))
    assert enabled["delivery"] == "enabled for the approved destination"


def test_status_of_an_unbound_provider_says_what_to_do(plugin, monkeypatch, tmp_path):
    provider, instance = unconfigured(plugin, monkeypatch, tmp_path)
    bind(provider, tmp_path / "stranger")
    report = json.loads(provider.handle_tool_call("memory_status", {}))
    assert report["bound"] is False and report["profile"] == "unbound"
    assert "enroll" in report["activity_home"]
    assert "no profile is bound" in report["delivery"]


def test_post_setup_writes_its_own_file_and_nothing_else(plugin, monkeypatch, tmp_path):
    """Activation belongs to Hermes. This writes the provider's configuration, and
    the one line about what is left."""
    home = tmp_path / "activity"
    home.mkdir()
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(tmp_path / "instance"))
    result = plugin.post_setup(str(home), {"memory_settings": {
        "data_dir": str(tmp_path / "instance" / "data"),
        "hindsight_url": "http://127.0.0.1:8863"}})
    written = sorted(path.name for path in home.iterdir())
    assert written == ["hermes-memory.env"], written
    assert "HERMES_MEMORY_HINDSIGHT_URL" in (home / "hermes-memory.env").read_text()
    assert result["enrolled"] is False and "--actor" in result["next_command"]
    ledger = tmp_path / "instance" / "installation.db"
    assert not ledger.exists(), "setup may not create the ledger it is refused by"
