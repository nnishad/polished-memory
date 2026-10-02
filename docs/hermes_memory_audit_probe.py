"""Offline audit probes: installed code, synthetic events, temporary SQLite only."""
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

HOST = Path('/home/jugaadu/codes/hermes-agent')
PLUGIN = Path('/home/jugaadu/.hermes/plugins/hermes-memory')
RELEASE = Path('/home/jugaadu/data/hermes-memory/runtime/current')
sys.path.insert(0, str(HOST))
os.environ['HERMES_MEMORY_RELEASE'] = str(RELEASE)

with tempfile.TemporaryDirectory(prefix='hermes-memory-audit-') as tmp:
    os.environ['HERMES_HOME'] = tmp
    for key in list(os.environ):
        if key.startswith('HERMES_MEMORY_') and key != 'HERMES_MEMORY_RELEASE':
            del os.environ[key]
    os.environ['HERMES_MEMORY_HOME'] = str(Path(tmp) / 'instance')
    os.environ['HERMES_MEMORY_ALLOWED_INFERENCE_HOSTS'] = '127.0.0.1'
    spec = importlib.util.spec_from_file_location(
        'audit_memory_plugin', PLUGIN / '__init__.py',
        submodule_search_locations=[str(PLUGIN)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    from audit_memory_plugin.provider import HermesMemoryProvider
    from audit_memory_plugin.spool import CaptureSpool
    from hermes_memory.sources.sdk import HermesEvents
    from tools.memory_tool import get_builtin_memory_store_flags, MemoryStore
    from hermes_memory.config import load_settings
    from hermes_memory.install.profiles import open_installation, ProfileRegistry
    instance_home = Path(tmp) / 'instance'
    instance_home.mkdir()
    registry = ProfileRegistry(open_installation(instance_home / 'installation.db'),
        root=instance_home, owner_principal='audit-owner', default_home=instance_home / 'data')
    proposal = registry.plan('default', Path(tmp))
    registry.enroll('default', Path(tmp), actor='audit-owner',
                    review_digest=proposal['review_digest'])
    registry.db.close()

    results = {}
    results['provider_store_native_flags'] = get_builtin_memory_store_flags({
        'memory': {'store': 'provider', 'provider': 'hermes-memory',
                   'memory_enabled': True, 'user_profile_enabled': True}})
    fresh = HermesMemoryProvider()
    results['fresh_provider_backup_paths'] = fresh.backup_paths()

    provider = HermesMemoryProvider()
    provider._session_id = 'synthetic-session'
    provider._spool = CaptureSpool(Path(tmp) / 'capture.db')
    provider.sync_turn('first question', 'first answer', messages=[{}, {}])
    provider.on_session_switch('synthetic-session', rewound=True)
    provider.sync_turn('different replacement question', 'different answer', messages=[{}, {}])
    rows = list(provider._spool.iter_all())
    results['rewound_same_length_turns'] = {
        'attempted': 2, 'stored': len(rows),
        'stored_user': json.loads(rows[0]['payload'])['user']}

    provider.on_delegation('synthetic task', 'synthetic result', child_session_id='child')
    provider.on_memory_write('remove', 'memory', '',
                             metadata={'previous_content': 'obsolete note'})
    provider.on_memory_write('replace', 'memory', 'new note',
                             metadata={'previous_content': 'old note'})
    adapter = HermesEvents(lambda after, limit: [])
    adapted = []
    for row in provider._spool.iter_all():
        event = {'event_id': row['event_id'], 'session_id': row['session_id'],
                 'created_at': row['created_at'], 'payload': json.loads(row['payload'])}
        _, envelopes, reason = adapter._one(event, set())
        adapted.append({'kind': event['payload']['kind'],
                        'action': event['payload'].get('action'),
                        'envelopes': len(envelopes or []), 'reason': reason,
                        'previous_content_preserved': bool(envelopes and any(
                            'previous_content' in e.get('metadata', {}) for e in envelopes))})
    results['event_adapter'] = adapted

    provider.on_memory_write('add', 'memory', 'repeatable fact', metadata={'session_id': 'one'})
    provider.on_memory_write('remove', 'memory', '', metadata={'previous_content': 'repeatable fact'})
    provider.on_memory_write('add', 'memory', 'repeatable fact', metadata={'session_id': 'two'})
    results['repeated_native_adds_stored'] = sum(
        json.loads(r['payload']).get('content') == 'repeatable fact'
        for r in provider._spool.iter_all())

    provider.on_session_end([{'role': 'user', 'content': 'long body ' + 'x' * 5000,
                              'author': {'id': 'alice'}}])
    tail = list(provider._spool.iter_all())[-1]
    payload = json.loads(tail['payload'])
    results['session_end'] = {
        'input_chars': 5010, 'stored_chars': len(payload['messages'][0]['text']),
        'truncation_marked': 'truncated' in payload,
        'author_preserved': 'author' in payload['messages'][0]}
    before = len(list(provider._spool.iter_all()))
    provider.on_pre_compress([
        {'role': 'user', 'content': [{'type': 'text', 'text': 'remember this caption'}]}
    ], require_checkpoint=True)
    results['multimodal_strict_checkpoint'] = {
        'returned_successfully': True,
        'events_added': len(list(provider._spool.iter_all())) - before}
    results['setup_hindsight_url_default'] = next(
        f['default'] for f in fresh.get_config_schema() if f['key'] == 'hindsight_url')
    fresh.save_config({'hindsight_url': 'http://127.0.0.1:9999',
                       'data_dir': str(Path(tmp) / 'unapproved-data')}, tmp)
    from audit_memory_plugin.client import bind
    activity = bind(tmp)
    loaded = activity.settings
    results['saved_profile_config_effect'] = {
        'profile_file_written': (Path(tmp) / 'hermes-memory.json').exists(),
        'configured_url': 'http://127.0.0.1:9999', 'effective_url': loaded.hindsight_url,
        'enrollment_data_dir_preserved': loaded.data_dir == instance_home / 'data'}
    activity.close()
    provider._spool.close()
    print(json.dumps(results, indent=2))
    assert results['provider_store_native_flags'] == (False, False)
    assert results['fresh_provider_backup_paths']
    assert results['rewound_same_length_turns']['stored'] == 2
    assert all(not item['reason'] and item['envelopes'] for item in adapted)
    assert all(item['previous_content_preserved'] for item in adapted
               if item['kind'] == 'native_memory_write')
    assert results['repeated_native_adds_stored'] == 2
    assert results['session_end']['stored_chars'] == results['session_end']['input_chars']
    assert results['session_end']['author_preserved']
    assert results['multimodal_strict_checkpoint']['events_added'] == 1
    assert results['setup_hindsight_url_default'] == 'http://127.0.0.1:8888'
    assert results['saved_profile_config_effect']['effective_url'] == 'http://127.0.0.1:9999'
    assert results['saved_profile_config_effect']['enrollment_data_dir_preserved']
