"""Real host/provider integration against two temporary homes; no network or live state.

Run using the Hermes interpreter, passing --hermes-source to its editable checkout.
"""
import argparse
import importlib.util
import json
import os
import sys
import tempfile
import weakref
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hermes-source", type=Path, required=True)
    parser.add_argument("--installed-plugin", type=Path)
    parser.add_argument("--release", type=Path)
    args = parser.parse_args()
    framework = Path(__file__).resolve().parents[1]
    package_path = (next(args.release.glob("lib/python*/site-packages"))
                    if args.release else framework / "src")
    sys.path[:0] = [str(args.hermes_source), str(package_path)]
    with tempfile.TemporaryDirectory(prefix="memory-gateway-probe-") as scratch:
        root = Path(scratch)
        for key in list(os.environ):
            if key.startswith("HERMES_MEMORY_"):
                os.environ.pop(key)
        os.environ.update(HERMES_MEMORY_HOME=str(root / "instance"),
                          HERMES_MEMORY_OWNER_PRINCIPAL="synthetic-owner",
                          HERMES_MEMORY_INFERENCE_ENABLED="false",
                          HERMES_MEMORY_DELIVERY_ENABLED="true",
                          HERMES_MEMORY_DELIVERY_TARGET="telegram:12345")
        if args.release:
            os.environ["HERMES_MEMORY_RELEASE"] = str(args.release)
        from hermes_memory.config import load_settings
        from hermes_memory.install.profiles import ProfileRegistry
        from hermes_memory.proactive.inquiries import InquiryStore
        from hermes_memory.proactive.policy import AttentionPolicy
        from hermes_memory.prospective.goals import GoalStore
        from hermes_memory.storage.evidence import EvidenceStore
        from agent.memory_manager import MemoryManager
        from agent.turn_context import _memory_turn_start_and_prefetch
        from gateway.config import Platform
        from gateway.inbound_message_context import context_for_event
        from gateway.session_identity import RoutingIdentity, _IDENTITY_ATTR
        settings = load_settings()
        registry = ProfileRegistry.open(settings)
        plugin_path = args.installed_plugin or framework / "integrations" / "hermes-memory"
        spec = importlib.util.spec_from_file_location("gateway_probe_plugin", plugin_path / "__init__.py",
                                                      submodule_search_locations=[str(plugin_path)])
        plugin = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = plugin
        spec.loader.exec_module(plugin)
        providers, records = {}, {}
        for profile in ("a", "b"):
            home = root / profile
            plan = registry.plan(profile, home)
            registry.enroll(profile, home, actor=settings.owner_principal,
                            review_digest=plan["review_digest"])
            provider = plugin.HermesMemoryProvider()
            assert provider.is_available()
            provider.initialize("session-" + profile, hermes_home=str(home), platform="telegram",
                                chat_id="12345", user_id="12345")
            with provider._open_store() as store:
                questions = InquiryStore(store, owner_principal=settings.owner_principal)
                questions.allow_replies(actor=settings.owner_principal, on=True, reason="Synthetic probe only")
                AttentionPolicy(store, owner_principal=settings.owner_principal).configure(
                    actor=settings.owner_principal, timezone_name="UTC", quiet_from="00:00", quiet_until="00:01",
                    cooldown_minutes=0, max_immediate_per_day=10, shadow=False)
                goal = GoalStore(store, owner_principal=settings.owner_principal).propose(
                    title="Synthetic " + profile, statement="Distinct synthetic plan " + profile,
                    proposed_by="agent:probe")["id"]
                inquiry = questions.ask(decision="goal-activation", subject_id=goal,
                                        question="Activate this synthetic profile-specific plan?")["id"]
                def transport(body):
                    return {"sent": True, "platform": "telegram", "chat_id": "12345", "message_id": "100"}
                transport.hermes_home = str(home)
                assert questions.send_next(sink=transport, destination="telegram:12345", holder="probe")["sent"] == 1
            providers[profile], records[profile] = provider, (goal, inquiry)
        class Adapter:
            pass
        adapter = Adapter()
        checks = []
        for profile in ("a", "b", "a"):
            home = root / profile
            source = SimpleNamespace(platform=Platform.TELEGRAM, chat_id="12345", user_id="12345",
                                     chat_type="dm", thread_id=None, is_bot=False)
            setattr(source, _IDENTITY_ATTR, RoutingIdentity(profile, profile, home, home,
                                                            transport=weakref.ref(adapter)))
            event = SimpleNamespace(text="yes", message_id="101", reply_to_message_id="100", internal=False)
            envelope = context_for_event(event, source)
            manager = MemoryManager()
            manager.add_provider(providers[profile])
            agent = SimpleNamespace(_memory_manager=manager, _user_turn_count=1, session_id="session-" + profile)
            context = _memory_turn_start_and_prefetch(agent, "yes", {"id": "12345"}, envelope)
            assert records[profile][1] in context
            other = "b" if profile == "a" else "a"
            assert records[other][1] not in context
            with providers[profile]._open_store() as store:
                assert GoalStore(store).get(records[profile][0]).status == "active"
            checks.append({"profile": profile, "context": True, "state": "active"})
        for provider in providers.values():
            provider.shutdown()
        registry.db.close()
        print(json.dumps({"ok": True, "network_calls": 0, "checks": checks}))


if __name__ == "__main__":
    main()
