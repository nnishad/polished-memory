"""Loader helpers for plugin tests."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

INTEGRATIONS = Path(__file__).resolve().parents[1] / "integrations"
# Hermes names a plugin by its directory, so this path and PROVIDER_NAME must agree.
PLUGIN = INTEGRATIONS / "hermes-memory"


def load_plugin(name: str = "hm_plugin"):
    """Import the plugin package from its directory.

    The plugin is loaded by Hermes's own directory importer at runtime; tests
    must not depend on the repository layout being on sys.path.
    """
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, PLUGIN / "__init__.py", submodule_search_locations=[str(PLUGIN)]
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
