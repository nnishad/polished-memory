# Hermes memory-related production source inventory

Generated 2026-10-01 against Hermes revision `f97608f178d1ffeca59860195ab7da295f7c8e5f`.

This is a lexical discovery inventory, not a claim that every file was read or every branch tested. It excludes test files and restricts matching to provider/manager/store interfaces, native memory filenames, session search, and checkpoint/backup lifecycle vocabulary. Runtime RAM management is outside scope. Newly added or dynamically referenced code may not match.

Search roots: `agent`, `tools`, `gateway`, `hermes_cli`, `tui_gateway`, `plugins`, `cron`, `acp_adapter`, `ui-tui/src`, `apps/desktop/src`, `apps/desktop/electron`.

## acp_adapter

- [acp_adapter/commands.py](/home/jugaadu/codes/hermes-agent/acp_adapter/commands.py)
- [acp_adapter/entry.py](/home/jugaadu/codes/hermes-agent/acp_adapter/entry.py)
- [acp_adapter/session.py](/home/jugaadu/codes/hermes-agent/acp_adapter/session.py)
- [acp_adapter/tools.py](/home/jugaadu/codes/hermes-agent/acp_adapter/tools.py)

## agent

- [agent/agent_init.py](/home/jugaadu/codes/hermes-agent/agent/agent_init.py)
- [agent/agent_runtime_helpers.py](/home/jugaadu/codes/hermes-agent/agent/agent_runtime_helpers.py)
- [agent/anthropic_adapter.py](/home/jugaadu/codes/hermes-agent/agent/anthropic_adapter.py)
- [agent/aux_accounting.py](/home/jugaadu/codes/hermes-agent/agent/aux_accounting.py)
- [agent/background_review.py](/home/jugaadu/codes/hermes-agent/agent/background_review.py)
- [agent/context_breakdown.py](/home/jugaadu/codes/hermes-agent/agent/context_breakdown.py)
- [agent/context_compressor.py](/home/jugaadu/codes/hermes-agent/agent/context_compressor.py)
- [agent/conversation_compression.py](/home/jugaadu/codes/hermes-agent/agent/conversation_compression.py)
- [agent/display.py](/home/jugaadu/codes/hermes-agent/agent/display.py)
- [agent/inline_tool_executors.py](/home/jugaadu/codes/hermes-agent/agent/inline_tool_executors.py)
- [agent/learning_graph.py](/home/jugaadu/codes/hermes-agent/agent/learning_graph.py)
- [agent/learning_mutations.py](/home/jugaadu/codes/hermes-agent/agent/learning_mutations.py)
- [agent/memory_manager.py](/home/jugaadu/codes/hermes-agent/agent/memory_manager.py)
- [agent/memory_provider.py](/home/jugaadu/codes/hermes-agent/agent/memory_provider.py)
- [agent/prompt_builder.py](/home/jugaadu/codes/hermes-agent/agent/prompt_builder.py)
- [agent/side_question.py](/home/jugaadu/codes/hermes-agent/agent/side_question.py)
- [agent/system_prompt.py](/home/jugaadu/codes/hermes-agent/agent/system_prompt.py)
- [agent/tool_dispatch_helpers.py](/home/jugaadu/codes/hermes-agent/agent/tool_dispatch_helpers.py)
- [agent/tool_executor.py](/home/jugaadu/codes/hermes-agent/agent/tool_executor.py)
- [agent/tool_guardrails.py](/home/jugaadu/codes/hermes-agent/agent/tool_guardrails.py)
- [agent/tool_result_classification.py](/home/jugaadu/codes/hermes-agent/agent/tool_result_classification.py)
- [agent/transports/hermes_tools_mcp_server.py](/home/jugaadu/codes/hermes-agent/agent/transports/hermes_tools_mcp_server.py)
- [agent/turn_context.py](/home/jugaadu/codes/hermes-agent/agent/turn_context.py)
- [agent/turn_summary.py](/home/jugaadu/codes/hermes-agent/agent/turn_summary.py)
- [agent/turn_tool_round.py](/home/jugaadu/codes/hermes-agent/agent/turn_tool_round.py)

## apps

- [apps/desktop/src/api/system.ts](/home/jugaadu/codes/hermes-agent/apps/desktop/src/api/system.ts)
- [apps/desktop/src/app/chat/composer/inline-refs.ts](/home/jugaadu/codes/hermes-agent/apps/desktop/src/app/chat/composer/inline-refs.ts)
- [apps/desktop/src/app/settings/config-field.tsx](/home/jugaadu/codes/hermes-agent/apps/desktop/src/app/settings/config-field.tsx)
- [apps/desktop/src/app/settings/config-settings.tsx](/home/jugaadu/codes/hermes-agent/apps/desktop/src/app/settings/config-settings.tsx)
- [apps/desktop/src/app/settings/constants.ts](/home/jugaadu/codes/hermes-agent/apps/desktop/src/app/settings/constants.ts)
- [apps/desktop/src/app/settings/helpers.ts](/home/jugaadu/codes/hermes-agent/apps/desktop/src/app/settings/helpers.ts)
- [apps/desktop/src/app/settings/memory/connect.tsx](/home/jugaadu/codes/hermes-agent/apps/desktop/src/app/settings/memory/connect.tsx)
- [apps/desktop/src/app/settings/memory/field-control.tsx](/home/jugaadu/codes/hermes-agent/apps/desktop/src/app/settings/memory/field-control.tsx)
- [apps/desktop/src/app/settings/memory/provider-config-modal.tsx](/home/jugaadu/codes/hermes-agent/apps/desktop/src/app/settings/memory/provider-config-modal.tsx)
- [apps/desktop/src/app/settings/memory/provider-config-panel.tsx](/home/jugaadu/codes/hermes-agent/apps/desktop/src/app/settings/memory/provider-config-panel.tsx)
- [apps/desktop/src/components/assistant-ui/tool/fallback-model/index.ts](/home/jugaadu/codes/hermes-agent/apps/desktop/src/components/assistant-ui/tool/fallback-model/index.ts)
- [apps/desktop/src/components/assistant-ui/tool/run-summary.ts](/home/jugaadu/codes/hermes-agent/apps/desktop/src/components/assistant-ui/tool/run-summary.ts)
- [apps/desktop/src/hermes.ts](/home/jugaadu/codes/hermes-agent/apps/desktop/src/hermes.ts)
- [apps/desktop/src/i18n/ar.ts](/home/jugaadu/codes/hermes-agent/apps/desktop/src/i18n/ar.ts)
- [apps/desktop/src/i18n/de.ts](/home/jugaadu/codes/hermes-agent/apps/desktop/src/i18n/de.ts)
- [apps/desktop/src/i18n/en.ts](/home/jugaadu/codes/hermes-agent/apps/desktop/src/i18n/en.ts)
- [apps/desktop/src/i18n/es.ts](/home/jugaadu/codes/hermes-agent/apps/desktop/src/i18n/es.ts)
- [apps/desktop/src/i18n/fr.ts](/home/jugaadu/codes/hermes-agent/apps/desktop/src/i18n/fr.ts)
- [apps/desktop/src/i18n/ja.ts](/home/jugaadu/codes/hermes-agent/apps/desktop/src/i18n/ja.ts)
- [apps/desktop/src/i18n/ru.ts](/home/jugaadu/codes/hermes-agent/apps/desktop/src/i18n/ru.ts)
- [apps/desktop/src/i18n/types.ts](/home/jugaadu/codes/hermes-agent/apps/desktop/src/i18n/types.ts)
- [apps/desktop/src/i18n/zh-hant.ts](/home/jugaadu/codes/hermes-agent/apps/desktop/src/i18n/zh-hant.ts)
- [apps/desktop/src/i18n/zh.ts](/home/jugaadu/codes/hermes-agent/apps/desktop/src/i18n/zh.ts)
- [apps/desktop/src/lib/session-refs.ts](/home/jugaadu/codes/hermes-agent/apps/desktop/src/lib/session-refs.ts)
- [apps/desktop/src/types/hermes.ts](/home/jugaadu/codes/hermes-agent/apps/desktop/src/types/hermes.ts)

## cron

- [cron/scheduler.py](/home/jugaadu/codes/hermes-agent/cron/scheduler.py)

## gateway

- [gateway/platforms/api_server_memory_sessions.py](/home/jugaadu/codes/hermes-agent/gateway/platforms/api_server_memory_sessions.py)
- [gateway/run.py](/home/jugaadu/codes/hermes-agent/gateway/run.py)
- [gateway/run_agent_cache.py](/home/jugaadu/codes/hermes-agent/gateway/run_agent_cache.py)
- [gateway/run_notifications.py](/home/jugaadu/codes/hermes-agent/gateway/run_notifications.py)
- [gateway/run_shutdown.py](/home/jugaadu/codes/hermes-agent/gateway/run_shutdown.py)
- [gateway/run_turn.py](/home/jugaadu/codes/hermes-agent/gateway/run_turn.py)
- [gateway/session.py](/home/jugaadu/codes/hermes-agent/gateway/session.py)
- [gateway/slash_commands.py](/home/jugaadu/codes/hermes-agent/gateway/slash_commands.py)

## hermes_cli

- [hermes_cli/agent_import.py](/home/jugaadu/codes/hermes-agent/hermes_cli/agent_import.py)
- [hermes_cli/backup.py](/home/jugaadu/codes/hermes-agent/hermes_cli/backup.py)
- [hermes_cli/claw.py](/home/jugaadu/codes/hermes-agent/hermes_cli/claw.py)
- [hermes_cli/cli_commands_mixin.py](/home/jugaadu/codes/hermes-agent/hermes_cli/cli_commands_mixin.py)
- [hermes_cli/cli_session_mixin.py](/home/jugaadu/codes/hermes-agent/hermes_cli/cli_session_mixin.py)
- [hermes_cli/cli_shutdown.py](/home/jugaadu/codes/hermes-agent/hermes_cli/cli_shutdown.py)
- [hermes_cli/codex_runtime_switch.py](/home/jugaadu/codes/hermes-agent/hermes_cli/codex_runtime_switch.py)
- [hermes_cli/config_defaults.py](/home/jugaadu/codes/hermes-agent/hermes_cli/config_defaults.py)
- [hermes_cli/doctor_state.py](/home/jugaadu/codes/hermes-agent/hermes_cli/doctor_state.py)
- [hermes_cli/kanban_db.py](/home/jugaadu/codes/hermes-agent/hermes_cli/kanban_db.py)
- [hermes_cli/main_agent_cmds.py](/home/jugaadu/codes/hermes-agent/hermes_cli/main_agent_cmds.py)
- [hermes_cli/memory_provider_migration.py](/home/jugaadu/codes/hermes-agent/hermes_cli/memory_provider_migration.py)
- [hermes_cli/memory_setup.py](/home/jugaadu/codes/hermes-agent/hermes_cli/memory_setup.py)
- [hermes_cli/observability/shared_metrics_contract.py](/home/jugaadu/codes/hermes-agent/hermes_cli/observability/shared_metrics_contract.py)
- [hermes_cli/oneshot.py](/home/jugaadu/codes/hermes-agent/hermes_cli/oneshot.py)
- [hermes_cli/plugin_python_deps.py](/home/jugaadu/codes/hermes-agent/hermes_cli/plugin_python_deps.py)
- [hermes_cli/plugins.py](/home/jugaadu/codes/hermes-agent/hermes_cli/plugins.py)
- [hermes_cli/plugins_cmd.py](/home/jugaadu/codes/hermes-agent/hermes_cli/plugins_cmd.py)
- [hermes_cli/plugins_manifest.py](/home/jugaadu/codes/hermes-agent/hermes_cli/plugins_manifest.py)
- [hermes_cli/profile_memory_config.py](/home/jugaadu/codes/hermes-agent/hermes_cli/profile_memory_config.py)
- [hermes_cli/profiles.py](/home/jugaadu/codes/hermes-agent/hermes_cli/profiles.py)
- [hermes_cli/prompt_size.py](/home/jugaadu/codes/hermes-agent/hermes_cli/prompt_size.py)
- [hermes_cli/subcommands/memory.py](/home/jugaadu/codes/hermes-agent/hermes_cli/subcommands/memory.py)
- [hermes_cli/tips.py](/home/jugaadu/codes/hermes-agent/hermes_cli/tips.py)
- [hermes_cli/tools_config.py](/home/jugaadu/codes/hermes-agent/hermes_cli/tools_config.py)
- [hermes_cli/web_models.py](/home/jugaadu/codes/hermes-agent/hermes_cli/web_models.py)
- [hermes_cli/web_routers/dashboard_ui.py](/home/jugaadu/codes/hermes-agent/hermes_cli/web_routers/dashboard_ui.py)
- [hermes_cli/web_routers/memory_providers.py](/home/jugaadu/codes/hermes-agent/hermes_cli/web_routers/memory_providers.py)
- [hermes_cli/web_routers/ops.py](/home/jugaadu/codes/hermes-agent/hermes_cli/web_routers/ops.py)
- [hermes_cli/web_server.py](/home/jugaadu/codes/hermes-agent/hermes_cli/web_server.py)
- [hermes_cli/web_server_config.py](/home/jugaadu/codes/hermes-agent/hermes_cli/web_server_config.py)

## plugins

- [plugins/disk-cleanup/disk_cleanup.py](/home/jugaadu/codes/hermes-agent/plugins/disk-cleanup/disk_cleanup.py)
- [plugins/memory/__init__.py](/home/jugaadu/codes/hermes-agent/plugins/memory/__init__.py)
- [plugins/memory/byterover/__init__.py](/home/jugaadu/codes/hermes-agent/plugins/memory/byterover/__init__.py)
- [plugins/memory/holographic/__init__.py](/home/jugaadu/codes/hermes-agent/plugins/memory/holographic/__init__.py)
- [plugins/memory/honcho/__init__.py](/home/jugaadu/codes/hermes-agent/plugins/memory/honcho/__init__.py)
- [plugins/memory/honcho/cli.py](/home/jugaadu/codes/hermes-agent/plugins/memory/honcho/cli.py)
- [plugins/memory/honcho/dialectic.py](/home/jugaadu/codes/hermes-agent/plugins/memory/honcho/dialectic.py)
- [plugins/memory/honcho/session.py](/home/jugaadu/codes/hermes-agent/plugins/memory/honcho/session.py)
- [plugins/memory/honcho/session_migration.py](/home/jugaadu/codes/hermes-agent/plugins/memory/honcho/session_migration.py)
- [plugins/memory/honcho/tool_schemas.py](/home/jugaadu/codes/hermes-agent/plugins/memory/honcho/tool_schemas.py)
- [plugins/memory/mem0/__init__.py](/home/jugaadu/codes/hermes-agent/plugins/memory/mem0/__init__.py)
- [plugins/memory/mem0/_setup.py](/home/jugaadu/codes/hermes-agent/plugins/memory/mem0/_setup.py)
- [plugins/memory/openviking/__init__.py](/home/jugaadu/codes/hermes-agent/plugins/memory/openviking/__init__.py)
- [plugins/memory/openviking/_setup.py](/home/jugaadu/codes/hermes-agent/plugins/memory/openviking/_setup.py)
- [plugins/memory/retaindb/__init__.py](/home/jugaadu/codes/hermes-agent/plugins/memory/retaindb/__init__.py)
- [plugins/memory/supermemory/__init__.py](/home/jugaadu/codes/hermes-agent/plugins/memory/supermemory/__init__.py)

## tools

- [tools/daemon_pool.py](/home/jugaadu/codes/hermes-agent/tools/daemon_pool.py)
- [tools/delegate_tool.py](/home/jugaadu/codes/hermes-agent/tools/delegate_tool.py)
- [tools/delegate_tool_results.py](/home/jugaadu/codes/hermes-agent/tools/delegate_tool_results.py)
- [tools/delegate_tool_toolsets.py](/home/jugaadu/codes/hermes-agent/tools/delegate_tool_toolsets.py)
- [tools/file_tools_write_guards.py](/home/jugaadu/codes/hermes-agent/tools/file_tools_write_guards.py)
- [tools/mcp_tool_agent.py](/home/jugaadu/codes/hermes-agent/tools/mcp_tool_agent.py)
- [tools/memory_tool.py](/home/jugaadu/codes/hermes-agent/tools/memory_tool.py)
- [tools/memory_tool_store.py](/home/jugaadu/codes/hermes-agent/tools/memory_tool_store.py)
- [tools/session_search_tool.py](/home/jugaadu/codes/hermes-agent/tools/session_search_tool.py)
- [tools/skill_manager_tool.py](/home/jugaadu/codes/hermes-agent/tools/skill_manager_tool.py)
- [tools/write_approval.py](/home/jugaadu/codes/hermes-agent/tools/write_approval.py)

## tui_gateway

- [tui_gateway/contracts/tools_mcp_plugins.py](/home/jugaadu/codes/hermes-agent/tui_gateway/contracts/tools_mcp_plugins.py)
- [tui_gateway/methods_tools.py](/home/jugaadu/codes/hermes-agent/tui_gateway/methods_tools.py)
- [tui_gateway/session_lifecycle.py](/home/jugaadu/codes/hermes-agent/tui_gateway/session_lifecycle.py)

