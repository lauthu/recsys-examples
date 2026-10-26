# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RunConfigs for the tests, as the runner writes them for the container."""

from bench.config import (
    Model,
    Resources,
    RunConfig,
    Runs,
    Scenario,
    Solution,
    Source,
    Workload,
)

WORKLOAD = Workload(
    context_length=4, beam_width=4, output_tokens=3, concurrency=1, seed=0
)


def run_config(
    engine="sid-gr",
    *,
    workload=WORKLOAD,
    runs=Runs(rounds=1, requests=3, warmups=2, profile_requests=1),
    dtype="bfloat16",
    env=None,
    server_cpus=(4, 5),
    client_cpus=(6,),
    catalog_items=None,
):
    model = Model(
        id="Qwen/Qwen3-0.6B",
        revision="e" * 40,
        sha256={"config.json": "f" * 64},
        dtype=dtype,
    )
    solution = Solution(
        engine=engine,
        source=Source(url="https://github.com/example/engine", commit="c" * 40),
        env=dict(env or {}),
        server_settings=None if engine == "sid-gr" else {"page_size": 1},
    )
    return RunConfig(
        scenario=Scenario(model=model, workload=workload, runs=runs),
        solution=solution,
        resources=Resources(list(server_cpus), list(client_cpus), "0"),
        vocab_size=5000,
        catalog_items=catalog_items,
    )
