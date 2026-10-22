# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Declared patches: how the launchers (server_*.py) change an engine's server.

Each launcher lists every method it patches in one PATCHES table, so the full
difference from the engine's native server is in one place. Every hook follows
the same rules:

- it calls the original with the same arguments and returns its result unchanged;
- it only counts calls, reads state, or pushes NVTX ranges;
- it never synchronizes the GPU.

apply() installs the entries for this run kind and for the code paths the
engine has (its adapter's `features`). A missing target stops the server, so a
new engine revision cannot silently skip a patch. Each applied patch is printed
to the server log.

configuration() builds the GET /pilot/config snapshot that both launchers serve:
the server's own settings and its environment, made JSON-safe, with the text
values of secret-looking names redacted.
"""

import functools
import importlib
import types
from typing import Callable, NamedTuple

# A name is secret if one of its words (split at "_", "-" and ".") is one of these.
SECRET_WORDS = frozenset({"TOKEN", "SECRET", "PASSWORD", "KEY"})


class Patch(NamedTuple):
    target: str  # "package.module:Owner.method" or "package.module:function"
    hook: Callable  # hook(original) -> replacement
    adds: str  # what the patch adds, for readers and the server log
    profile_only: bool = False  # only in the profiled process (BENCH_PROFILE=1)
    # The adapter feature (Adapter.features) the target exists for; None: always.
    needs: str | None = None


def wrap(owner, name, hook, target):
    original = getattr(owner, name, None)
    if not callable(original):
        raise RuntimeError(f"unsupported engine revision: {target} is missing")
    setattr(owner, name, functools.wraps(original)(hook(original)))


def resolve(target):
    """("package.module:Owner.method") -> (Owner, "method")."""
    module, _, attribute = target.partition(":")
    owner = importlib.import_module(module)
    *parents, name = attribute.split(".")
    for parent in parents:
        owner = getattr(owner, parent)
    return owner, name


def selected(patches, features, profile):
    return [
        patch
        for patch in patches
        if (profile or not patch.profile_only)
        and (patch.needs is None or patch.needs in features)
    ]


def apply(patches, features, profile):
    """Install the selected patches and return them."""
    chosen = selected(patches, features, profile)
    for patch in chosen:
        wrap(*resolve(patch.target), patch.hook, patch.target)
        kind = "profile" if patch.profile_only else "always"
        print(f"[bench] patched {patch.target} ({kind}): {patch.adds}", flush=True)
    return chosen


def plain(value, depth=0):
    """JSON-safe copy of a configuration value: scalars, lists, mappings and the
    public attributes of objects; anything else, or anything nested deeper, as text."""
    if getattr(type(value), "_member_map_", None) is not None:  # an Enum member
        return plain(value.value, depth)
    if value is None or isinstance(value, bool):
        return value
    for kind in (int, str):  # exact types, not subclasses
        if isinstance(value, kind):
            return kind(value)
    if isinstance(value, float):
        return value if abs(value) < float("inf") else str(value)  # NaN, inf: text
    nested = depth < 4 and not isinstance(value, (type, types.ModuleType))
    if nested and callable(getattr(value, "items", None)):
        return {str(k): plain(v, depth + 1) for k, v in value.items()}
    if nested and isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=str) if isinstance(value, (set, frozenset)) else value
        return [plain(item, depth + 1) for item in items]
    if nested and hasattr(value, "__dict__") and not callable(value):
        return {
            k: plain(v, depth + 1)
            for k, v in vars(value).items()
            if not k.startswith("_")
        }
    return str(value)


def secret(name):
    words = str(name).upper().replace("-", "_").replace(".", "_").split("_")
    return not SECRET_WORDS.isdisjoint(words)


def redacted(mapping):
    """The mapping, with the text values of secret names replaced at any depth.
    Only text can be a secret: numbers such as bos_token_id or num_key_value_heads
    stay."""
    return {
        k: (
            "redacted"
            if secret(k) and isinstance(v, str) and v
            else redacted(v)
            if isinstance(v, dict)
            else v
        )
        for k, v in mapping.items()
    }


def configuration(settings, source, environment):
    """GET /pilot/config: the server's settings, named by where they come from, and
    its environment."""
    return {
        "source": source,
        "settings": redacted(plain(settings)),
        "environment": redacted(dict(sorted(environment.items()))),
    }
