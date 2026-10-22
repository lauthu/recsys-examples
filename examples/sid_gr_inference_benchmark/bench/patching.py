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
"""

import functools
import importlib
from typing import Callable, NamedTuple


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
