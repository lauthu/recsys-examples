# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapter: everything the benchmark needs to know about an engine, as logic.

An adapter holds no settings. On the host it checks a solution against the
scenario and writes the engine's catalog file; in the container it reads the
RunConfig (bench.config), which holds both, and says how to start the
engine's server, call it, read its beams and check what it reports. The
runner, the client, the launchers and the reports ask the adapter instead of
branching on the engine name. Standard library only: adapters run on the host
and in the container.
"""

from bench.common import CACHE_ENV

# Engine settings the container takes only from the RunConfig, never from the image.
AMBIENT = ("GR_", "SGLANG_", "MODEL_")


class Adapter:
    name: str  # the engine name a solution gives as `engine`
    title: str
    source_label: str  # what a solution's source checkout must be, for errors
    entry: str  # a file that must exist in the source checkout
    # How the image is built: a Dockerfile under images/, or, for a dev branch
    # built from source by images/build.sh, the engine's own recipe.
    recipe: str
    mount = "/source"  # where the source checkout appears in the container
    fingerprint: tuple[str, ...] = (
        "examples/sid-gr-inference",
        "corelib/gr_decode_atten",
    )
    # Optional solution sections this engine requires; it rejects the others.
    sections: tuple[str, ...] = ()
    # Environment the runner or the adapter sets; a solution must not.
    managed_env = frozenset(
        CACHE_ENV
        + "HOME PYTHONPATH CUDA_VISIBLE_DEVICES BENCH_PROFILE BENCH_ENGINE".split()
    )
    # The engine's code paths that launcher patches target (patching.Patch.needs).
    features: frozenset[str] = frozenset()
    ready = "/pilot/ready"
    # With a catalog, the engine serves a generated copy of the checkpoint.
    model_view = False
    catalog_supported = True
    # The catalog's native file, which the runner generates; None: the JSONL.
    catalog_file: str | None = None

    @property
    def catalog_source(self):
        """Where the server reads the catalog in the container."""
        if self.model_view:
            return f"/model/{self.catalog_file}"
        return f"/work/{self.catalog_file or 'catalog.jsonl'}"

    # ---- host ------------------------------------------------------------
    def validate(self, solution, scenario):
        """Engine-specific checks of a Solution against the Scenario; raise ValueError."""

    def convert_catalog(self, paths, model, vocab_size):
        """Bytes of the engine's native catalog file (only if catalog_file is set);
        `model` is the scenario's Model."""
        raise NotImplementedError

    def overlaps_steps(self, solution):
        """Whether the engine launches a step before the previous one's results
        are processed (SGLang's overlap scheduler); then steps cannot be told
        apart in a trace."""
        return False

    # ---- container -------------------------------------------------------
    def server(self, config, port, profile):
        """(argv, env) that start the server for a RunConfig. env holds what the
        benchmark sets; the client adds it to the container's environment
        without AMBIENT."""
        raise NotImplementedError

    def server_args(self, config, port):
        """SGLang ServerArgs this adapter sets over the solution's server_settings;
        None for an engine configured only through its environment."""
        return None

    def environment(self, config, profile):
        # CPU thread pools match the server CPUs unless the solution says otherwise.
        threads = str(len(config.resources.server_cpus))
        env = {"OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads}
        # What the launcher marks: "requests" or "steps"; "0" when not profiled.
        env.update(config.solution.env, BENCH_PROFILE=profile or "0")
        return env

    def profile(self, port, start):
        """Start or stop the cudaProfilerApi capture range in the server process."""
        raise NotImplementedError

    def payload(self, row, workload):
        raise NotImplementedError

    def beams(self, response, rid):
        """[(token_ids, cumulative logprob)] in the server's order."""
        raise NotImplementedError

    def catalog_status(self, port, runtime):
        raise NotImplementedError

    def check_source(self, runtime):
        """Raise if the server is not running the mounted source."""

    def check_counters(self, config, counters, requests, runtime):
        """Raise if the measured requests did not take the requested code path."""

    # ---- reports ---------------------------------------------------------
    def graph_usage(self, rounds):
        """How the measured rounds used CUDA graphs, from their counters."""
        raise NotImplementedError

    def setup_rows(self, solution, catalog):
        """Extra (label, text) rows for the report's setup table."""
        return []


def measured_requests(result):
    """Measured requests of one round's result, over every concurrency."""
    return sum(len(level["latency_ms"]) for level in result["levels"])


def sum_counters(rounds):
    """The rounds' counters added up; a counter no round reports is 0."""
    totals = Counts()
    for result in rounds:
        for key, value in result["counters"].items():
            totals[key] += value
    return totals


class Counts(dict):
    def __missing__(self, key):
        return 0
