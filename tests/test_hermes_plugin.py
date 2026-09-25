"""C13 provider contract and the durability guarantee the host cannot give us."""
from __future__ import annotations

import json
import os

import pytest
from hermes_memory.storage.evidence import EvidenceError, EvidenceStore

from plugin_loader import INTEGRATIONS, PLUGIN, load_plugin


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


@pytest.fixture()
def provider(plugin, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(tmp_path / "profile"))
    instance = plugin.HermesMemoryProvider()
    assert instance.is_available(), instance.unavailable_reason()
    instance.initialize("sess-1", hermes_home=str(tmp_path / "profile"), platform="cli")
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
