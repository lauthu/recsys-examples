# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The engines. Each family has an adapter, which tells the runner and the
client how to configure, start, call and check its engines (adapter.py), and
a launcher, which runs the engine's own HTTP server in the container with the
patches it lists in PATCHES (patching.py):

    sglang.py  sglang_server.py   public SGLang, beam-search-new, vedantjh2-sglang
    sid_gr.py  sid_gr_server.py   SID-GR, gr-inference-internal

This package loads no adapter until get() asks for one: the launchers run in
the engine's server process, which should load only what it needs.
"""

import functools
import importlib

# A solution's `engine` -> "module:adapter class" in this package.
ENGINES = {
    "sid-gr": "sid_gr:SidGrAdapter",
    "gr-inference-internal": "sid_gr:GRInferenceInternalAdapter",
    "sglang": "sglang:PublicSGLangAdapter",
    "sglang-beam-search-new": "sglang:BeamSearchNewAdapter",
    "vedantjh2-sglang": "sglang:Vedantjh2SGLangAdapter",
}


@functools.cache
def get(engine):
    """The adapter of an engine name, as a solution gives it in `engine`; one
    instance per engine, since adapters hold no state."""
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {', '.join(ENGINES)}")
    module, _, name = ENGINES[engine].partition(":")
    return getattr(importlib.import_module(f"{__name__}.{module}"), name)()
