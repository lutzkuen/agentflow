"""Backward-compatible import alias for the renamed ``tokenclaw`` package."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys as _sys

import tokenclaw as _tokenclaw
from tokenclaw import *  # noqa: F401,F403


class _AgentflowProxyAliasLoader(importlib.abc.Loader):
    def create_module(self, spec):
        target_name = _target_name(spec.name)
        module = importlib.import_module(target_name)
        _sys.modules[spec.name] = module
        return module

    def exec_module(self, module) -> None:
        return None


class _AgentflowProxyAliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("agentflow_proxy."):
            return None
        target_name = _target_name(fullname)
        target_spec = importlib.util.find_spec(target_name)
        if target_spec is None:
            return None
        is_package = target_spec.submodule_search_locations is not None
        spec = importlib.util.spec_from_loader(
            fullname,
            _AgentflowProxyAliasLoader(),
            origin=target_spec.origin,
            is_package=is_package,
        )
        if spec is not None and is_package:
            spec.submodule_search_locations = target_spec.submodule_search_locations
        return spec


def _target_name(fullname: str) -> str:
    return "tokenclaw" + fullname[len("agentflow_proxy") :]


if not any(isinstance(finder, _AgentflowProxyAliasFinder) for finder in _sys.meta_path):
    _sys.meta_path.insert(0, _AgentflowProxyAliasFinder())

_sys.modules[__name__] = _tokenclaw
