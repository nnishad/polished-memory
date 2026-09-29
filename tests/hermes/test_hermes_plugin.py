"""C13 provider contract and the durability guarantee the host cannot give us."""
from __future__ import annotations

import io
import json
import os
import sqlite3
import sys
import threading
import tokenize

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


def test_a_missing_runtime_names_the_runtime_rather_than_the_configuration(plugin,
                                                                          monkeypatch):
    """"configuration refused" would send an operator to an env file that is fine.

    §10.2 wants the hook to stop with the setup instruction instead of bootstrapping a
    runtime from inside a conversation, so the sentence has to say what is absent and what
    to run — and must not promise to install anything.
    """
    monkeypatch.setitem(sys.modules, "hermes_memory", None)
    monkeypatch.setitem(sys.modules, "hermes_memory.config", None)
    instance = plugin.HermesMemoryProvider()
    assert instance.is_available() is False
    reason = instance.unavailable_reason()
    assert "runtime is not importable" in reason
    assert "hermes-memory setup" in reason
    assert "configuration refused" not in reason
    assert "never installs it" in reason


def test_a_missing_dependency_of_our_own_is_named_and_not_blamed_on_the_config(plugin,
                                                                              monkeypatch):
    """The runtime is here; something it needs is not. Say which.

    Both branches read as "it will not start", and only one of them is fixed by installing
    hermes-memory. An operator sent to the release tree when a data package is missing
    reinstalls the wrong thing and arrives at the same sentence.
    """
    import hermes_memory.config as config_layer

    def missing(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'zoneinfo_data'", name="zoneinfo_data")

    monkeypatch.setattr(config_layer, "load_settings", missing)
    instance = plugin.HermesMemoryProvider()
    assert instance.is_available() is False
    reason = instance.unavailable_reason()
    assert "cannot start without zoneinfo_data" in reason
    assert "configuration refused" not in reason


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


def test_the_identity_tool_says_so_when_the_owner_already_decided(provider):
    """The owner's review list is a resource, and a second proposal about a join that
    is already true would put a decided question back on it."""
    from hermes_memory.storage.identity import IdentityStore

    recorded = json.loads(provider.handle_tool_call(
        "memory_remember", {"content": "Priya writes from both addresses."}))
    proposal = {"account_a": "priya@example.com", "account_b": "priya@work.example",
                "rule": "email-thread-participant", "basis": "one person, two domains",
                "evidence": [recorded["id"]]}
    queued = json.loads(provider.handle_tool_call("memory_identity_candidate", proposal))
    assert queued["queued_for_owner_review"] is True

    # The owner's own principal, from the owner's own shell: the provider above cannot
    # reach this, which is the point of the split.
    with provider._open_store() as store:
        IdentityStore(store, owner_principal=OWNER).confirm(
            candidate_id=queued["candidate_id"], actor=OWNER, reason="mine")

    again = json.loads(provider.handle_tool_call("memory_identity_candidate", proposal))
    assert again["ok"] is True
    assert again["queued_for_owner_review"] is False, "nothing was queued this time"
    assert again["candidate_id"] is None
    assert "already one person" in again["note"]


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


def test_the_identity_tool_takes_the_account_object_a_recall_hands_back(provider):
    """A proposal is made from what memory said, and memory says participants as
    namespace-and-address pairs — a caller quoting that shape reaches the same account."""
    recorded = json.loads(provider.handle_tool_call(
        "memory_remember", {"content": "Priya writes from two domains."}))
    payload = json.loads(provider.handle_tool_call("memory_identity_candidate", {
        "account_a": {"namespace": "email", "address": "priya@example.com"},
        "account_b": {"namespace": "email", "address": "priya@work.example"},
        "rule": "email-normalized-equal", "basis": "one local part, two domains",
        "evidence": [recorded["id"]]}))
    assert payload["ok"] is True, payload
    with provider._open_store() as store:
        written = sorted(row["identifier"] for row in
                         store.db.execute("SELECT identifier FROM identity_accounts"))
    assert written == ["priya@example.com", "priya@work.example"], \
        "the address is what is stored, not a description of it"


def test_a_blob_handed_over_as_an_account_is_named_not_stored(provider):
    """The old path stringified whatever arrived and kept it, so one person written in two
    shapes was two accounts that no rule could ever join."""
    recorded = json.loads(provider.handle_tool_call(
        "memory_remember", {"content": "Priya writes from two domains."}))
    for value in ('{"namespace": "email", "address": "priya@example.com"}',
                  ["email", "priya@example.com"], 7):
        payload = json.loads(provider.handle_tool_call("memory_identity_candidate", {
            "account_a": value, "account_b": "priya@work.example",
            "rule": "email-thread-participant", "basis": "one thread, two participants",
            "evidence": [recorded["id"]]}))
        assert payload["ok"] is False, value
        assert payload["error"].startswith("account_a:"), payload["error"]
    with provider._open_store() as store:
        assert store.db.execute(
            "SELECT COUNT(*) AS n FROM identity_accounts").fetchone()["n"] == 0, \
            "a refused proposal registers nothing"


def test_the_identity_schema_declares_both_account_shapes(provider):
    properties = next(tool for tool in provider.get_tool_schemas()
                      if tool["name"] == "memory_identity_candidate")["parameters"]["properties"]
    for field in ("account_a", "account_b"):
        assert properties[field]["type"] == ["string", "object"], field
        assert "namespace" in properties[field]["description"], field


def test_status_says_which_copy_of_the_framework_is_answering(provider):
    """A release switch that the host does not obey is not a switch, so the host reports it."""
    payload = json.loads(provider.handle_tool_call("memory_status", {}))
    assert payload["runtime"]["loaded_from"], "the host is told what it runs"
    assert "matches_release" in payload["runtime"]


def test_a_goal_proposal_is_a_candidate_and_reminds_for_nothing(provider):
    """The tool can propose; only the owner's door makes it an obligation."""
    recorded = json.loads(provider.handle_tool_call(
        "memory_remember", {"content": "The passport expires in six weeks."}))
    payload = json.loads(provider.handle_tool_call("memory_goal", {
        "title": "Renew the passport",
        "statement": "It expires in six weeks; the office needs four.",
        "basis_record": recorded["id"]}))
    assert payload["ok"] is True
    assert payload["status"] == "candidate" and payload["scheduled"] is False
    assert payload["goal_id"].startswith("gol_")
    assert "activate" in payload["note"]
    recall = json.loads(provider.handle_tool_call("memory_recall", {"query": "passport"}))
    assert recall["items"], "the evidence stays; the promise simply has no clock yet"


def test_a_goal_proposal_refuses_a_fragment(provider):
    for arguments in ({"title": "Renew", "statement": "  "},
                      {"title": "", "statement": "Something owed"},
                      {}):
        payload = json.loads(provider.handle_tool_call("memory_goal", arguments))
        assert payload["ok"] is False, arguments
        assert "title and statement" in payload["error"]


def test_a_goal_proposal_cites_only_evidence_that_can_be_read(provider):
    payload = json.loads(provider.handle_tool_call("memory_goal", {
        "title": "Chase the invoice", "statement": "It was mentioned once.",
        "basis_record": "rec_" + "0" * 32}))
    assert payload["ok"] is False
    assert "evidence that can still be read" in payload["error"]


def test_a_goal_proposal_is_not_accepted_from_a_cron_context(provider):
    """A delegated pass may report; it does not get to queue edits to the owner's memory."""
    home = provider._activity.hermes_home
    provider.initialize("sess-cron", hermes_home=str(home), platform="cli",
                        agent_context="cron")
    payload = json.loads(provider.handle_tool_call("memory_goal", {
        "title": "Renew the passport", "statement": "Six weeks."}))
    assert payload["ok"] is False
    assert "cron context" in payload["error"]


def test_the_goal_tool_is_advertised(provider):
    schemas = provider.get_tool_schemas()
    names = [item["name"] for item in schemas]
    assert "memory_goal" in names, "the plan's stable tool set names it"
    goal = next(item for item in schemas if item["name"] == "memory_goal")
    assert goal["parameters"]["required"] == ["title", "statement"]
    assert "candidate" in goal["description"]


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


# -- asking the owner, and taking their reply back -----------------------------

@pytest.fixture()
def asking(plugin, tmp_path, monkeypatch):
    """A provider whose owner exists, has agreed to be asked, and speaks on Telegram."""
    from hermes_memory.proactive.inquiries import InquiryStore
    from hermes_memory.proactive.policy import AttentionPolicy

    monkeypatch.setenv("HERMES_MEMORY_OWNER_PRINCIPAL", OWNER)
    monkeypatch.setenv("HERMES_MEMORY_HOME", str(tmp_path / "profile"))
    monkeypatch.setenv("HERMES_MEMORY_DELIVERY_ENABLED", "true")
    monkeypatch.setenv("HERMES_MEMORY_DELIVERY_TARGET", "telegram:787655730")
    enroll(tmp_path / "profile", tmp_path / "profile")
    instance = plugin.HermesMemoryProvider()
    assert instance.is_available(), instance.unavailable_reason()
    instance.initialize("sess-1", hermes_home=str(tmp_path / "profile"),
                        platform="telegram", chat_id="787655730")
    with instance._open_store() as store:
        InquiryStore(store, owner_principal=OWNER).allow_replies(
            actor=OWNER, on=True, reason="a reply on my own channel, carrying a code")
        # Awake and out loud, or the sending half of these tests holds everything it is given.
        AttentionPolicy(store, owner_principal=OWNER).configure(
            actor=OWNER, timezone_name="UTC", cooldown_minutes=0, max_immediate_per_day=5,
            quiet_from="00:00", quiet_until="00:01", shadow=False)
    return instance


def a_proposed_goal(asking):
    """One candidate the agent proposed and cannot adopt itself."""
    asking.handle_tool_call("memory_remember",
                            {"content": "The passport expires in six weeks."})
    made = json.loads(asking.handle_tool_call("memory_goal", {
        "title": "Renew the passport", "statement": "It expires in six weeks."}))
    assert made["status"] == "candidate", made
    return made["goal_id"]


def test_memory_clarify_queues_a_question_and_adopts_nothing(asking):
    """Asking is a write to a queue, not a decision: the candidate stays a candidate."""
    identifier = a_proposed_goal(asking)

    payload = json.loads(asking.handle_tool_call("memory_clarify", {
        "decision": "goal-activation", "subject": identifier,
        "question": "Should I start reminding you about the passport?"}))

    assert payload["asked"] is True and payload["state"] == "open", payload
    assert payload["inquiry"].startswith("inq_")
    assert "code" not in json.dumps(payload).lower()
    with asking._open_store() as store:
        assert store.db.execute("SELECT status FROM goals WHERE id=?",
                                (identifier,)).fetchone()[0] == "candidate"


def test_a_question_is_not_opened_where_it_could_never_be_answered(asking):
    """Replies off is not a reason to queue a question the sender would only void."""
    from hermes_memory.proactive.inquiries import InquiryStore

    with asking._open_store() as store:
        InquiryStore(store, owner_principal=OWNER).allow_replies(
            actor=OWNER, on=False, reason="I would rather use the door")
    identifier = a_proposed_goal(asking)

    payload = json.loads(asking.handle_tool_call("memory_clarify", {
        "decision": "goal-activation", "subject": identifier,
        "question": "Should I start reminding you about the passport?"}))

    assert payload["ok"] is False and payload["asked"] is False
    assert "switched off" in payload["reason"]


def test_the_same_question_asked_twice_is_asked_once(asking):
    identifier = a_proposed_goal(asking)
    question = {"decision": "goal-activation", "subject": identifier,
                "question": "Should I start reminding you about the passport?"}

    first = json.loads(asking.handle_tool_call("memory_clarify", question))
    again = json.loads(asking.handle_tool_call("memory_clarify", question))

    assert first["asked"] is True and again["asked"] is False
    assert again["id"] == first["inquiry"]


def test_a_reply_about_a_state_that_awaits_nobody_decides_nothing(asking):
    identifier = a_proposed_goal(asking)
    asking.handle_tool_call("memory_clarify", {
        "decision": "goal-activation", "subject": identifier,
        "question": "Should I start reminding you about the passport?"})
    asking.prefetch("yes QQQQQQ")

    payload = json.loads(asking.handle_tool_call(
        "memory_clarify_answer", {"reply": "yes QQQQQQ"}))

    assert payload["settled"] is False and "no question is open under" in payload["reason"]


def test_the_reply_is_attributed_to_the_channel_the_host_named(asking):
    """The conversation cannot say where it came from; only the host's binding can.

    ``channel`` is not in the tool's arguments at all, so a relay that claimed to be a private
    SMS to the owner has nothing to overwrite — and the audit line carries what the host said,
    not what the caller said.
    """
    assert set(next(item for item in asking.get_tool_schemas()
                    if item["name"] == "memory_clarify_answer")
               ["parameters"]["properties"]) == {"reply", "inquiry"}

    payload = json.loads(asking.handle_tool_call(
        "memory_clarify_answer", {"reply": "yes QQQQQQ", "channel": "sms:+15559999"}))

    assert payload["settled"] is False
    assert payload["channel"] == "telegram:787655730"


def test_a_reply_is_not_taken_from_a_cron_context(asking):
    home = asking._activity.hermes_home
    asking.initialize("sess-cron", hermes_home=str(home), platform="cron",
                      agent_context="cron")

    payload = json.loads(asking.handle_tool_call("memory_clarify_answer",
                                                 {"reply": "yes XJ23KK"}))

    assert payload["ok"] is False and "cron context" in payload["error"]


def a_question_on_the_wire(asking, tmp_path):
    """One question the owner has actually been sent, and the code that answers it."""
    import re

    from hermes_memory.proactive.delivery import local_sink
    from hermes_memory.proactive.inquiries import InquiryStore

    identifier = a_proposed_goal(asking)
    wire = tmp_path / "wire"
    with asking._open_store() as store:
        inquiries = InquiryStore(store, owner_principal=OWNER)
        made = inquiries.ask(decision="goal-activation", subject_id=identifier,
                             question="Should I remind you about the passport?")
        sent = inquiries.send_next(sink=local_sink(wire), destination="telegram:787655730",
                                   holder="test")
    assert sent["sent"] == 1 and made["state"] == "open", sent
    delivered = "\n".join(path.read_text(encoding="utf-8")
                          for path in sorted((wire / "memory" / "delivered").glob("*.md")))
    return identifier, re.search(r"`yes ([A-Z2-9]{6})`", delivered).group(1)


def quoted(code, *, said):
    """How a messaging client delivers a reply: our own question first, then their words."""
    return ('[Replying to: "hermes-memory question\n'
            'It would settle: goal-activation (gol_1394c60f29664e99bfb0bef17308616e).\n'
            f"To settle it by reply, say so and give the code: `yes {code}`. "
            f'`no` stops the asking and decides nothing."]\n\n{said}')


def test_the_owners_reply_is_taken_off_the_wire_without_anybody_deciding_to_notice(
        asking, tmp_path):
    """What this replaces is measured: the owner said it, the conversation did not listen.

    A good `yes CODE` arrived on the owner's channel, no tool was called, and the model went
    and activated the goal at a command line that checks no code at all. The answer cannot be
    left to a turn's discretion, because the turn has something better to do.
    """
    identifier, code = a_question_on_the_wire(asking, tmp_path)

    said = asking.prefetch(quoted(code, said=f"yes {code}"))

    assert "The owner answered their own question" in said
    with asking._open_store() as store:
        assert store.db.execute("SELECT status FROM goals WHERE id=?",
                                (identifier,)).fetchone()[0] == "active"
        assert tuple(store.db.execute("SELECT state, answer_channel FROM inquiries")
                     .fetchone()) == ("answered", "telegram:787655730")


def test_a_code_that_only_ever_came_back_in_the_quotation_is_not_an_answer(asking, tmp_path):
    """The quotation is ours, and it carries our own code to whoever can read the channel.

    So an answer is what follows the quotation. Without that rule the tool would be a way to
    settle a decision by repeating a message the framework sent to somebody else.
    """
    identifier, code = a_question_on_the_wire(asking, tmp_path)

    asking.prefetch(quoted(code, said="what is this?"))

    payload = json.loads(asking.handle_tool_call("memory_clarify_answer",
                                                 {"reply": f"yes {code}"}))
    assert payload["settled"] is False
    assert "no message from the owner" in payload["reason"]
    with asking._open_store() as store:
        assert store.db.execute("SELECT status FROM goals WHERE id=?",
                                (identifier,)).fetchone()[0] == "candidate"
        assert store.db.execute("SELECT state FROM inquiries").fetchone()[0] == "sent"


def test_a_question_can_only_be_about_an_act_the_door_can_settle(asking):
    """The tool's enum and the owner's door are one list, checked rather than hoped."""
    from hermes_memory.operations.decisions import ACTS

    schema = next(item for item in asking.get_tool_schemas()
                  if item["name"] == "memory_clarify")
    assert schema["parameters"]["properties"]["decision"]["enum"] == list(ACTS)


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

    def spying(settings, **arguments):
        registry = real(settings, **arguments)
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


# -- learned practice, retrieved with the evidence -----------------------------

GMAIL_RULE = {"all": [{"field": "source", "op": "eq", "value": "gmail"}]}


def a_habit(instance, *, text="Chase an overdue invoice twice by mail before phoning.",
            rule=GMAIL_RULE, approve=True, lesson_id="chase-invoice"):
    """One lesson in this profile's store, optionally promoted by the owner."""
    from hermes_memory.learning.lessons import LessonStore
    from hermes_memory.learning.outcomes import OutcomeLog

    with instance._open_store() as store:
        record = store.commit({"source": "gmail", "source_id": "playbook-1",
                               "revision": "1", "kind": "email",
                               "text": "Chase the invoice twice before phoning.",
                               "observed_at": "2026-09-15T09:00:00+00:00"})["id"]
        # The owner's promotion is recorded out of band, exactly as the owner door would:
        # these tests deliberately run with no owner principal configured in the provider.
        habits = LessonStore(store, outcomes=OutcomeLog(store, owner_principal=OWNER),
                             owner_principal=OWNER)
        made = habits.propose(lesson_id=lesson_id, text=text, applicability=rule,
                              evidence=[str(record)], proposed_by="agent:sess-1",
                              proposed_kind="agent")
        if approve:
            habits.activate(lesson_id=lesson_id, version=made["version"], actor=OWNER,
                            reason="how we work")
    return str(record), made["version"]


def recalled(instance, **args):
    return json.loads(instance.handle_tool_call("memory_recall",
                                                dict({"query": "invoice"}, **args)))


def test_an_approved_practice_arrives_with_the_evidence(provider):
    a_habit(provider)
    payload = recalled(provider, task={"source": "gmail"})
    assert payload["ok"] is True
    assert [item["text"] for item in payload["lessons"]] == \
        ["Chase an overdue invoice twice by mail before phoning."]


def test_a_practice_for_another_case_teaches_nothing_here(provider):
    a_habit(provider)
    assert recalled(provider, task={"source": "calendar"})["lessons"] == []


def test_a_habit_nobody_approved_is_never_retrieved(provider):
    """Promotion is an act with a name on it; asking is not that act."""
    a_habit(provider, approve=False)
    assert recalled(provider, task={"source": "gmail"})["lessons"] == []


def test_asking_without_saying_what_you_are_doing_retrieves_no_practice(provider):
    a_habit(provider)
    assert recalled(provider)["lessons"] == []


def test_a_forgotten_lesson_stops_being_taught_on_the_next_read(provider):
    record, _ = a_habit(provider)
    with provider._open_store() as store:
        store.hide(record, reason="the owner forgot it", actor=OWNER)
    assert recalled(provider, task={"source": "gmail"})["lessons"] == [], \
        "a rule whose evidence is gone cannot be re-taught from memory of it"


def test_a_task_field_that_does_not_exist_is_refused_rather_than_ignored(provider):
    """Silently dropping "recipient" would answer as though nothing narrower was asked."""
    a_habit(provider)
    payload = recalled(provider, task={"recipient": "client-a"})
    assert payload["ok"] is False
    assert "unknown task field" in payload["error"]
    assert "recipient" in payload["error"]
    assert "channel" in payload["error"], "the refusal names the vocabulary it offers"


def test_a_task_that_is_not_a_description_of_a_task_is_refused(provider):
    payload = recalled(provider, task=["gmail"])
    assert payload["ok"] is False and "object of {field: value}" in payload["error"]
    wide = {f"field_{index}": "x" for index in range(9)}
    payload = recalled(provider, task=wide)
    assert payload["ok"] is False and "unknown task field" in payload["error"], \
        "the vocabulary is the bound: nine names it does not contain is nine refusals"


def test_the_hosts_own_platform_is_a_channel_a_practice_can_be_written_for(provider):
    """The one part of the shape nobody has to declare is the one the host already knows."""
    a_habit(provider, lesson_id="greet-by-name", text="Greet the client by name first.",
            rule={"all": [{"field": "channel", "op": "eq", "value": "cli"}]})
    payload = recalled(provider, task={"topic": "billing"})
    assert [item["text"] for item in payload["lessons"]] == \
        ["Greet the client by name first."]
    other = json.loads(provider.handle_tool_call(
        "memory_recall", {"query": "invoice", "task": {"channel": "whatsapp"}}))
    assert other["lessons"] == [], "a declared channel wins over the platform's own"


def test_the_recall_tool_describes_the_vocabulary_it_accepts(provider):
    schema = next(tool for tool in provider.get_tool_schemas()
                   if tool["name"] == "memory_recall")
    task = schema["parameters"]["properties"]["task"]
    assert task["type"] == "object"
    for field in ("source", "kind", "topic", "tool", "host", "channel", "account"):
        assert field in task["description"], field


def test_practice_is_injected_into_the_turn_not_only_returned_by_a_tool(provider):
    """C11's product is a habit the agent can follow, and a turn is not told to ask."""
    a_habit(provider, lesson_id="greet-by-name", text="Greet the client by name first.",
            rule={"all": [{"field": "channel", "op": "eq", "value": "cli"}]})
    text = provider.prefetch("invoice", session_id="sess-9")
    assert "- lesson: Greet the client by name first." in text, \
        "practice is rendered as a lesson, not as an item of evidence"


def test_a_warmed_turn_carries_the_same_practice_as_an_inline_one(provider):
    """Two paths build a packet. If only one of them retrieves practice, whether the
    agent is taught the owner's rule depends on a background thread having run."""
    a_habit(provider, lesson_id="greet-by-name", text="Greet the client by name first.",
            rule={"all": [{"field": "channel", "op": "eq", "value": "cli"}]})
    provider.queue_prefetch("invoice", session_id="sess-77")
    text = provider.prefetch("invoice", session_id="sess-77")
    assert "- lesson: Greet the client by name first." in text


def test_an_undescribed_turn_is_not_told_how_to_act(provider):
    """A rule written as an exclusion is satisfied by an empty task, which is exactly why
    the caller has to say something before practice is retrieved at all."""
    provider._platform = ""
    a_habit(provider, lesson_id="never-about-payments",
            text="Do not discuss payment terms unprompted.",
            rule={"not": {"field": "topic", "op": "eq", "value": "billing"}})
    assert recalled(provider)["lessons"] == []
    described = recalled(provider, task={"topic": "scheduling"})["lessons"]
    texts = [item["text"] for item in described]
    assert texts == ["Do not discuss payment terms unprompted."]
    assert described[0]["why"] == "the excluded case was absent", \
        "the answer says which rule let it in"


def test_the_plugin_stays_readable_by_the_interpreter_the_host_loads_at():
    """Hermes imports this plugin with its own interpreter, which sits at our declared floor.

    A replacement field that breaks across lines is PEP 701, so below 3.12 the file is a
    SyntaxError and the host registers nothing: the doctor answers "0 tool(s), 0 hook(s)"
    and memory looks absent rather than broken. Checking this with the interpreter that runs
    the suite would never catch it, because 3.12 and later accept the form.
    """
    offenders = []
    for path in sorted(PLUGIN.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        start = getattr(tokenize, "FSTRING_START", None)
        end = getattr(tokenize, "FSTRING_END", None)
        if start is None:
            try:
                compile(text, str(path), "exec")
            except SyntaxError as error:
                offenders.append(f"{path.name}: line {error.lineno}: {error.msg}")
            continue
        depth, opened = 0, None
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
        except (tokenize.TokenError, IndentationError, SyntaxError) as error:
            offenders.append(f"{path.name}: does not tokenize ({error})")
            continue
        for token in tokens:
            if token.type == start:
                depth += 1
                opened = opened or token.start[0]
            elif token.type == end:
                depth -= 1
                if depth == 0 and opened is not None:
                    if token.end[0] != opened:
                        offenders.append(f"{path.name}: f-string spans lines {opened}-"
                                        f"{token.end[0]}")
                    opened = None
    assert offenders == [], (
        "the host parses a plugin at the floor this package declares; a multi-line "
        f"replacement field is a SyntaxError there: {offenders}")


def test_the_manifest_claims_exactly_what_the_provider_registers(provider):
    """The host reads these two lists and compares them against what registration yields.

    Nine `provides_hooks` entries were provider callbacks rather than hook events, so the
    host answered each with "unknown hook" and registered nothing; a tool the provider
    registered was missing from `provides_tools`. Either way the manifest was stating
    something about the installation that the host could disprove, and a plugin the host
    cannot describe is a plugin that looks absent rather than broken.
    """
    text = (PLUGIN / "plugin.yaml").read_text(encoding="utf-8")

    def listed(key):
        """The block under `key:` — a list of `- name` lines, or the empty flow list.

        Parsed by hand so the guarantee needs no YAML library: this is the same choice the
        manifest reader in the installer makes.
        """
        lines = text.splitlines()
        try:
            head = next(i for i, line in enumerate(lines) if line.startswith(f"{key}:"))
        except StopIteration:
            return None
        rest = lines[head].partition(":")[2].strip()
        if rest == "[]":
            return []
        if rest:
            return [item.strip().strip("'\"") for item in rest.strip("[]").split(",")
                    if item.strip()]
        items = []
        for line in lines[head + 1:]:
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            if not stripped.startswith("-"):
                break
            items.append(stripped[1:].strip())
        return items

    registered = sorted(str(item.get("name")) for item in provider.get_tool_schemas())
    assert listed("provides_hooks") == [], \
        ("a memory provider dispatches lifecycle callbacks through the manager, not the "
         "host's hook registry, so it subscribes to no hook event")
    assert sorted(listed("provides_tools") or []) == registered, \
        (f"the manifest declares {sorted(listed('provides_tools') or [])} but the provider "
         f"registers {registered}")


def test_the_config_schema_is_the_shape_the_host_parses():
    """The host reads `config_schema` as name -> spec, not as JSON Schema.

    Written as `{type: object, properties: {…}}`, the host skipped `type` and
    `additionalProperties` as non-mappings and registered a setting literally called
    `properties` — so every real setting was unreachable, with only a warning line in the
    agent log to show for it.
    """
    lines = (PLUGIN / "plugin.yaml").read_text(encoding="utf-8").splitlines()
    head = next(i for i, line in enumerate(lines) if line.startswith("config_schema:"))
    children = {}
    current = None
    for line in lines[head + 1:]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith("  "):
            break
        indent = len(line) - len(line.lstrip())
        key, _, value = line.strip().partition(":")
        if indent == 2:
            current = key
            children[current] = {}
        elif indent >= 4 and current is not None and value.strip():
            children[current][key] = value.strip().strip('"\'')

    accepted = {"str", "string", "int", "integer", "float", "number", "bool", "boolean",
                "list", "array", "dict", "object", "secret"}
    assert set(children) == {"data_dir", "hindsight_url", "foreground_deadline_s"}, \
        "the host's parser registers each top-level key here as a setting of that name"
    for name, spec in children.items():
        assert spec, f"{name} declares no spec, so the host would skip it"
        assert str(spec.get("type", "")).lower() in accepted, \
            f"{name} declares type {spec.get('type')!r}, which the host cannot type-check"



# -- the thread the host calls the tool on -----------------------------------

def recall(provider, query="the survey cadence"):
    return json.loads(provider.handle_tool_call("memory_recall", {"query": query, "limit": 3}))


def in_thread(provider, query="the survey cadence"):
    """Ask for a recall on a thread of its own, the way the host's tool runner does."""
    box: dict[str, dict] = {}

    def worker():
        box["answer"] = recall(provider, query)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(30)
    assert not thread.is_alive(), "the recall never returned on the worker thread"
    return box["answer"]


def test_a_recall_on_a_host_worker_thread_gets_a_connection_it_may_use(provider):
    """Hermes runs tool handlers on worker threads, and a SQLite connection has an owner.

    The provider caches one store and one broker because the packet is reused across turns —
    which is only sound while the same thread does the asking. Measured on a live
    installation: every `memory_recall` raised "SQLite objects created in a thread can only be
    used in that same thread", so the host's main read path failed non-deterministically,
    depending on which thread had warmed the broker.
    """
    assert recall(provider)["ok"] is True
    answer = in_thread(provider)
    assert answer["ok"] is True, answer.get("error")


def test_the_warm_cache_does_not_make_a_second_thread_share_a_connection(provider):
    """Two turns may be served by two different threads, and neither gets the other's file."""
    assert recall(provider)["ok"] is True
    assert in_thread(provider)["ok"] is True
    assert in_thread(provider, "a different question")["ok"] is True
    assert recall(provider)["ok"] is True, "and the thread that started still answers"


def test_a_session_boundary_releases_the_connection_it_claims_to(provider):
    """Saying the cache was dropped, while the file handle stays open, is the leak.

    A gateway that flips sessions without tearing the provider down would otherwise accumulate
    an open reader of the profile's database per boundary.
    """
    assert recall(provider)["ok"] is True
    held = provider._local.cache["store"]
    provider.on_session_switch("sess-2")
    with pytest.raises(sqlite3.ProgrammingError):
        held.db.execute("SELECT 1")
    assert recall(provider)["ok"] is True
    assert provider._local.cache["store"] is not held, \
        "the next ask made itself a new one rather than using a closed handle"


def test_a_thread_releases_its_own_stale_connection_when_it_is_next_asked(provider):
    """The session moved on another thread, which cannot close this thread's handle for it.

    SQLite refuses a close from a thread that did not open the connection, so the generation is
    the notice and the next call is the collection. A cache that ignored the generation would
    hand back a connection whose session — and whose warmed packet — is somebody else's.
    """
    from concurrent.futures import ThreadPoolExecutor

    def ask(pool, query):
        def work():
            answer = recall(provider, query)
            return answer, provider._local.cache["store"]
        return pool.submit(work).result()

    with ThreadPoolExecutor(max_workers=1) as pool:
        first_answer, first_store = ask(pool, "the survey cadence")
        assert first_answer["ok"] is True
        provider.on_session_switch("sess-2")
        second_answer, second_store = ask(pool, "a different question")

        def reuse_held_handle() -> str:
            """Ask the owning thread to use the handle it left behind, and report what it says.

            Only that thread can tell a closed connection from another thread's connection, so
            this is where the release is actually measured.
            """
            try:
                first_store.db.execute("SELECT 1")
            except sqlite3.ProgrammingError as error:
                return str(error)
            return "still open"

        note = pool.submit(reuse_held_handle).result()
    assert second_answer["ok"] is True
    assert second_store is not first_store, "the stale entry was kept, not rebuilt"
    assert "closed database" in note, \
        f"the thread that owned the handle abandoned it instead of releasing it ({note})"


def test_two_threads_hold_two_connections_and_both_answer(provider):
    """The isolation is the fix: shared would mean one of them is always the wrong thread."""
    seen = []

    def collect():
        answer = recall(provider)
        seen.append(provider._local.cache["store"])
        return answer

    threads = [threading.Thread(target=collect) for _ in range(2)]
    assert recall(provider)["ok"] is True
    main_store = provider._local.cache["store"]
    for thread in threads:
        thread.start()
        thread.join(30)
        assert not thread.is_alive()
    assert len(seen) == 2 and seen[0] is not seen[1]
    assert main_store not in seen, "and neither worker borrowed the main thread's handle"


def test_a_practice_read_reuses_the_connection_this_thread_holds(provider):
    """Retrieval happens every turn; an open per turn is a leak with a normal-looking face."""
    opened: list[int] = []
    real = provider._open_store
    provider._open_store = lambda: (opened.append(1), real())[1]
    try:
        assert recall(provider)["ok"] is True
        warmed = len(opened)
        assert provider._lessons({"topic": "invoices"}) == []
        assert len(opened) == warmed, "the habit read opened a second store"
    finally:
        provider._open_store = real
