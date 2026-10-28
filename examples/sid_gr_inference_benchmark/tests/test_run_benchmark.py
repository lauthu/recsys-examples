# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import dataclasses
import hashlib
import json
import random
import re
from argparse import Namespace

import pytest
import run_benchmark as runner
import yaml
from bench import config as configs
from bench import engines
from bench.common import read_json
from bench.config import Workload
from bench.engines.sglang import TRIE_HEAD_FILE, prefill_tokens
from bench.engines.sid_gr import server_options
from tests.fixtures import run_config

CONFIGS = (
    "sid_gr",
    "sglang",
    "sglang_beam_search_new",
    "vedantjh2_sglang",
    "gr_inference_internal",
)
SID, PUBLIC, LEGACY, VEDANTJH2, INTERNAL = CONFIGS
# The one solution tied to a workload: it captures graphs for one beam width.
GRAPHS = "vedantjh2_sglang_shared_graphs"
SCENARIOS = ("qwen3_0_6b", "qwen3_1_7b", "qwen3_4b", "qwen3_0_6b_load")


def path(name):
    kind = "scenarios" if name in SCENARIOS else "solutions"
    return runner.ROOT / "configs" / kind / f"{name}.yaml"


def raw(name):
    return yaml.safe_load(path(name).read_text())


def scenario(**changes):
    return override(raw("qwen3_0_6b"), changes)


def solution(name=SID, **changes):
    return override(raw(name), changes)


def shipped():
    return configs.load_scenario(path("qwen3_0_6b"))


def pinned(found, sha256="0" * 64):
    """The scenario, constrained to a catalog."""
    return dataclasses.replace(found, catalog=configs.CatalogPin(sha256=sha256))


def override(value, changes):
    """Apply dotted-path overrides such as {"workload.seed": 1}."""
    for dotted, item in changes.items():
        *parents, key = dotted.split(".")
        target = value
        for parent in parents:
            target = target[parent]
        target[key] = item
    return value


def write(tmp_path, value):
    file = tmp_path / "config.yaml"
    file.write_text(yaml.safe_dump(value))
    return file


def test_shipped_configs_load_into_typed_classes():
    for name in SCENARIOS:
        found = configs.load_scenario(path(name))
        assert isinstance(found.workload, configs.Workload)
        assert found.model.revision == raw(name)["model"]["revision"]
        assert found.model.dtype == "bfloat16" and found.catalog is None
        layout = configs.SidCodebooks(token_start=1000, codebook_size=1000)
        assert found.model.sid_codebooks == layout
    for name in CONFIGS:
        scenarios = [shipped()] + ([pinned(shipped())] if name != PUBLIC else [])
        for scenario_ in scenarios:
            adapter, found = configs.load_solution(path(name), scenario_)
            assert isinstance(found, configs.Solution) and adapter.name == found.engine
            assert isinstance(found.source, configs.Source)


def test_every_solution_runs_with_any_scenario(tmp_path):
    """Solutions hold no model or workload values."""
    larger = {"workload.beam_width": 512, "workload.concurrency": 8}
    larger["workload.output_tokens"] = 5
    scenarios = [configs.load_scenario(path(name)) for name in SCENARIOS]
    scenarios.append(configs.load_scenario(write(tmp_path, scenario(**larger))))
    load = {"workload.concurrency": [1, 2, 4, 8], "workload.min_context_length": 16}
    scenarios.append(configs.load_scenario(write(tmp_path, scenario(**load))))
    for found in scenarios:
        for name in CONFIGS:
            configs.load_solution(path(name), found)


INVALID_SCENARIO = [
    ({"extra": 1}, r"unknown keys \['extra'\]"),
    ({"workload.extra": 1}, r"scenario config.yaml.workload has unknown keys"),
    ({"cpus": {"server": [0], "client": [1]}}, r"unknown keys \['cpus'\]"),
    ({"workload.output_tokens": 1}, "workload.output_tokens must be >= 2"),
    ({"workload.beam_width": 1.5}, "workload.beam_width must be int, not float"),
    ({"workload.concurrency": 0}, "workload.concurrency must be >= 1"),
    ({"workload.concurrency": []}, "workload.concurrency must be >= 1"),
    ({"workload.concurrency": [2, 2]}, "a list of distinct such values"),
    ({"workload.concurrency": True}, "workload.concurrency must be int or list"),
    ({"workload.concurrency": [1, "2"]}, r"workload.concurrency\[1\] must be int"),
    ({"workload.min_context_length": 0}, "between 1 and context_length"),
    ({"workload.min_context_length": 4096}, "between 1 and context_length"),
    ({"runs.rounds": "3"}, "runs.rounds must be int, not str"),
    ({"runs.profile_concurrency": 2}, "must be one of workload.concurrency"),
    ({"model.revision": "main"}, "40-character Hub commit"),
    ({"model.dtype": "half"}, "model.dtype must be one of bfloat16, float16"),
    ({"catalog": {"sha256": "abc"}}, "catalog.sha256 must be a 64-character"),
    ({"model.sha256": {"model.safetensors": "0" * 64}}, "including config.json"),
    ({"model.sid_codebooks.codebook_size": 0}, "codebook_size must be >= 1"),
    ({"model.sid_codebooks.depth": 3}, "sid_codebooks has unknown keys"),
]


@pytest.mark.parametrize("changes, message", INVALID_SCENARIO)
def test_invalid_scenarios_fail_before_launch(tmp_path, changes, message):
    with pytest.raises(ValueError, match=message):
        configs.load_scenario(write(tmp_path, scenario(**changes)))


def test_a_scenario_with_a_missing_section_fails(tmp_path):
    value = scenario()
    del value["runs"]
    with pytest.raises(ValueError, match=r"missing keys \['runs'\]"):
        configs.load_scenario(write(tmp_path, value))


def test_yaml_duplicate_keys_and_syntax_errors_fail(tmp_path):
    file = tmp_path / "config.yaml"
    file.write_text(path("qwen3_0_6b").read_text() + "\nruns:\n  rounds: 1\n")
    with pytest.raises(ValueError, match=r"duplicate keys \['runs'\]"):
        configs.load_scenario(file)
    file.write_text("workload: [1,\n")
    with pytest.raises(ValueError, match="config.yaml"):
        configs.load_scenario(file)


INVALID_SOLUTION = [
    (SID, {"extra": 1}, r"unknown keys \['extra'\]"),
    (SID, {"env.GR_INFERENCE_PREFILL_SDPA_ENABLE_GQA": 1}, "must be str, not int"),
    (SID, {"env.GR_BEAM_WIDTH": "128"}, "runner-managed"),
    (SID, {"engine": "other"}, "engine must be one of"),
    (SID, {"source.commit": "c6eba91"}, "40-character"),
    (SID, {"source.url": "git@github.com:x/y.git"}, "https://"),
    (VEDANTJH2, {"source.branch": ""}, "source.branch must not be empty"),
    (SID, {"server_settings": {"tp_size": 1}}, r"sections \[\]"),
    (VEDANTJH2, {"trie_config": {"token_start": 0}}, r"unknown keys \['trie_config'\]"),
    (LEGACY, {"server_settings.enable_beam_search": False}, "enable_beam_search"),
    (LEGACY, {"server_settings.port": 30000}, "must not set"),
    (LEGACY, {"server_settings.tp_size": 2}, "tp_size"),
    (LEGACY, {"server_settings.disable_overlap_schedule": False}, "overlap"),
    (PUBLIC, {"server_settings.tp_size": 2}, "tp_size"),
    (PUBLIC, {"server_settings.cuda_graph_bs_decode": [256]}, "must not set"),
    (LEGACY, {"server_settings.cuda_graph_bs_prefill": [2048]}, "must not set"),
    (PUBLIC, {"server_settings.max_running_requests": 1280}, "must not set"),
    (PUBLIC, {"server_settings.dtype": "float16"}, "must not set"),
    (VEDANTJH2, {"server_settings.random_seed": 1}, "must not set"),
    (VEDANTJH2, {"env.SGLANG_PROFILE_V2": "1"}, "runner-managed"),
    (SID, {"env.PYTHON_BIN": "python3"}, "runner-managed"),
    (VEDANTJH2, {"server_settings.enable_beam_search": True}, "beam_width API"),
    (VEDANTJH2, {"env.SGLANG_BEAM_SHARED_CONTEXT_ATTENTION": "1"}, '"true"'),
    (VEDANTJH2, {"env.SGLANG_BEAM_TRIE_TOPK_LOGPROB": "1"}, '"true"'),
    (VEDANTJH2, {"env.SGLANG_BEAM_TRIE_DENSE_HEAD": "1"}, '"true"'),
    (VEDANTJH2, {"env.SGLANG_BEAM_TRIE_DENSE_HEAD": "true"}, "needs a catalog"),
    (VEDANTJH2, {"env.SGLANG_BEAM_SHARED_CONTEXT_WIDTH": "256"}, "graphs only with"),
    (INTERNAL, {"server_settings.port": 30000}, "must not set"),
    (INTERNAL, {"server_settings.decode_steps": 2}, "must not set"),
    (INTERNAL, {"server_settings.rec_vocab_file": "/tmp/sids"}, "must not set"),
    (INTERNAL, {"env.GR_CACHE_ROOT": "/tmp"}, "runner-managed"),
    (INTERNAL, {"env.GR_MAX_OUTPUT_TOKENS": "3"}, "runner-managed"),
]


@pytest.mark.parametrize("name, changes, message", INVALID_SOLUTION)
def test_invalid_solutions_fail_before_launch(tmp_path, name, changes, message):
    with pytest.raises(ValueError, match=message):
        configs.load_solution(write(tmp_path, solution(name, **changes)), shipped())


@pytest.mark.parametrize("name", [PUBLIC, VEDANTJH2])
def test_native_beam_search_may_use_the_overlap_scheduler(tmp_path, name):
    value = solution(name, **{"server_settings.disable_overlap_schedule": False})
    found = configs.load_solution(write(tmp_path, value), shipped())[1]
    assert found.server_settings["disable_overlap_schedule"] is False


def test_adapters_size_batches_for_the_requests_in_flight():
    workload = Workload(
        context_length=2048, beam_width=256, output_tokens=3, concurrency=3, seed=7
    )
    config = run_config("sglang", workload=workload)
    # Every beam holds a request slot; beam-search-new also keeps the request's.
    assert engines.get("sglang").server_args(config, 8000) == {
        "model_path": "/model",
        "host": "127.0.0.1",
        "port": 8000,
        "dtype": "bfloat16",
        "random_seed": 7,
        "cuda_graph_bs_decode": [256, 512, 768],
        "cuda_graph_bs_prefill": [2048, 4096, 6144],
        "max_running_requests": 3 * 257,
    }
    env = engines.get("sid-gr").server(config, 8000, profile=False)[1]
    assert env["GR_MAX_BATCH_SIZE"] == env["GR_BEAM_KV_POOL_CAPACITY"] == "3"
    assert env["GR_CONTEXT_KV_POOL_CAPACITY"] == "3"
    assert env["GR_DECODE_CUDA_GRAPH_BATCH_BUCKETS"] == "1,2,3"
    assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == "2"  # server CPUs
    # A solution may choose its own thread counts.
    config = run_config("sglang", workload=workload, env={"OMP_NUM_THREADS": "1"})
    env = engines.get("sglang").server(config, 8000, profile=False)[1]
    assert env["OMP_NUM_THREADS"] == "1" and env["MKL_NUM_THREADS"] == "2"


def test_one_deployment_serves_every_concurrency():
    workload = Workload(
        context_length=2048,
        beam_width=256,
        output_tokens=3,
        concurrency=[1, 3],
        seed=7,
    )
    config = run_config("vedantjh2-sglang", workload=workload)
    settings = engines.get("vedantjh2-sglang").server_args(config, 8000)
    assert settings["cuda_graph_bs_decode"] == [256, 512, 768]
    assert settings["max_running_requests"] == 3 * 257
    config = run_config("sid-gr", workload=workload)
    env = engines.get("sid-gr").server(config, 8000, profile=False)[1]
    assert env["GR_MAX_BATCH_SIZE"] == "3" and env["GR_CONTEXT_LEN"] == "2048"


def test_variable_lengths_get_prefill_graphs_within_one_and_a_half_times():
    fixed = Workload(
        context_length=2048, beam_width=4, output_tokens=3, concurrency=2, seed=0
    )
    variable = dataclasses.replace(fixed, min_context_length=256)
    sizes = prefill_tokens(variable, 2)
    assert sizes == [256, 384, 576, 864, 1296, 1944, 2916, 4096]
    assert all(b <= 1.5 * a for a, b in zip(sizes, sizes[1:]))
    assert prefill_tokens(fixed, 2) == [2048, 4096]


def test_sglang_adapters_set_only_the_settings_solutions_may_not():
    workload = Workload(
        context_length=8, beam_width=2, output_tokens=3, concurrency=1, seed=0
    )
    config = run_config(workload=workload, dtype="float16", catalog_items=4)
    for name in engines.ENGINES:
        adapter = engines.get(name)
        settings = adapter.server_args(config, 8000)
        assert settings is None or settings.keys() <= adapter.managed_settings
    legacy = engines.get("sglang-beam-search-new").server_args(config, 8000)
    assert legacy["beam_search_constraint_dict"] == "/work/catalog.pkl"


@pytest.mark.parametrize("name", [PUBLIC, INTERNAL])
def test_engines_with_server_options_need_server_settings(tmp_path, name):
    value = solution(name)
    del value["server_settings"]
    with pytest.raises(ValueError, match=r"sections \['server_settings'\]"):
        configs.load_solution(write(tmp_path, value), shipped())


def test_vedantjh2_trie_head_needs_three_tokens_and_the_sid_layout():
    found = shipped()
    four = dataclasses.replace(found.workload, output_tokens=4)
    four = dataclasses.replace(found, workload=four)
    configs.load_solution(path(VEDANTJH2), four)  # native beam search: any length
    with pytest.raises(ValueError, match="exactly three SID codebooks"):
        configs.load_solution(path(VEDANTJH2), pinned(four))
    plain = dataclasses.replace(found.model, sid_codebooks=None)
    plain = dataclasses.replace(found, model=plain)
    configs.load_solution(path(VEDANTJH2), plain)
    with pytest.raises(ValueError, match="needs model.sid_codebooks"):
        configs.load_solution(path(VEDANTJH2), pinned(plain))


def test_vedantjh2_ablations_load_with_a_catalog_and_the_beam_width(tmp_path):
    catalog = pinned(shipped())
    dense = {"env.SGLANG_BEAM_TRIE_DENSE_HEAD": "true"}
    configs.load_solution(write(tmp_path, solution(VEDANTJH2, **dense)), catalog)
    adapter, found = configs.load_solution(path(GRAPHS), catalog)
    assert adapter.name == "vedantjh2-sglang" and adapter.shared_graphs(found.env)
    wider = dataclasses.replace(catalog.workload, beam_width=512)
    for scenario_, changes in (
        (shipped(), {}),
        (dataclasses.replace(catalog, workload=wider), {}),
        (catalog, {"env.SGLANG_BEAM_SHARED_CONTEXT_WIDTH": "128"}),
        (catalog, {"env.SGLANG_BEAM_SHARED_CONTEXT_ATTENTION": "false"}),
    ):
        value = solution(GRAPHS, **changes)
        with pytest.raises(ValueError, match="graphs only with"):
            configs.load_solution(write(tmp_path, value), scenario_)


def test_sid_gr_serves_only_bfloat16():
    half = dataclasses.replace(shipped().model, dtype="float16")
    half = dataclasses.replace(shipped(), model=half)
    configs.load_solution(path(PUBLIC), half)  # SGLang takes the scenario's dtype
    for name in (SID, INTERNAL):
        with pytest.raises(ValueError, match="builds its model in bfloat16"):
            configs.load_solution(path(name), half)


def test_gr_inference_internal_takes_sid_grs_shape_and_its_options_after_it():
    workload = Workload(
        context_length=2048,
        beam_width=256,
        output_tokens=3,
        concurrency=[1, 2, 4, 8],
        seed=7,
    )
    config = run_config("gr-inference-internal", workload=workload)
    shipped_settings = raw(INTERNAL)["server_settings"]
    config = dataclasses.replace(
        config,
        solution=dataclasses.replace(config.solution, server_settings=shipped_settings),
    )
    argv, env = engines.get("gr-inference-internal").server(config, 8000, False)
    assert argv[:2] == ["bash", "/source/scripts/serve_qwen3_gr_http.sh"]
    assert argv[2:] == server_options(shipped_settings)
    assert argv[2:7] == [
        "--prefill-backend",
        "auto",
        "--decode-cuda-graph",
        "--no-prefill-cuda-graph",
        "--inflight-batching-mode",
    ]
    sid_gr = engines.get("sid-gr").server(config, 8000, profile=False)[1]
    shape = {key: sid_gr[key] for key in sid_gr if key != "GR_DECODE_ATTEN_ROOT"}
    assert shape.items() <= env.items() and env["GR_MAX_BATCH_SIZE"] == "8"
    assert env["GR_DECODE_ATTEN_ROOT"] == "/source/third_party/gr-decode-attention"
    assert env["GR_CACHE_ROOT"].startswith("/cache/")


def test_server_options_are_flags_and_values():
    settings = {"decode_cuda_graph": True, "prefill_cuda_graph": False}
    settings.update(decode_streams="1", buckets=[128, 256])
    assert server_options(settings) == [
        "--decode-cuda-graph",
        "--no-prefill-cuda-graph",
        "--decode-streams",
        "1",
        "--buckets",
        "128,256",
    ]


def test_any_checkpoint_dtype_loads(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_text("")
    for dtype in ("float16", "float32"):
        config = {"torch_dtype": dtype, "vocab_size": 5000}
        (model / "config.json").write_text(json.dumps(config))
        assert configs.load_model(model, shipped().workload) == config


def test_workload_is_seeded_and_skips_special_tokens(tmp_path):
    value = configs.load_scenario(
        write(tmp_path, scenario(**{"workload.context_length": 64}))
    )
    model = {"vocab_size": 50, "bos_token_id": 0, "eos_token_id": [1, 2]}
    rows = configs.make_workload(value, model)
    assert rows == configs.make_workload(value, model)
    assert len(rows) == 510 and rows[10]["request_id"] == "request-10"
    assert {t for row in rows for t in row["input_ids"]} <= set(range(3, 50))


def test_every_level_gets_new_requests_of_the_same_lengths(tmp_path):
    changes = {"workload.context_length": 64, "workload.min_context_length": 8}
    changes["workload.concurrency"] = [4, 1]
    value = configs.load_scenario(write(tmp_path, scenario(**changes)))
    model = {"vocab_size": 50, "bos_token_id": 0, "eos_token_id": [1, 2]}
    rows = configs.make_workload(value, model)
    # Each block: 10 warmups per client of concurrency 4, then 200 measured.
    warmups, size = configs.block_layout(value)
    first, second = rows[:size], rows[size:]
    assert warmups == 40 and len(rows) == 2 * size == 1080
    assert len({row["request_id"] for row in rows}) == len(rows)
    assert all(a["input_ids"] != b["input_ids"] for a, b in zip(first, second))
    assert [len(r["input_ids"]) for r in first] == [len(r["input_ids"]) for r in second]


def test_variable_prompt_lengths_are_seeded_and_bounded(tmp_path):
    changes = {"workload.context_length": 64, "workload.min_context_length": 8}
    value = configs.load_scenario(write(tmp_path, scenario(**changes)))
    model = {"vocab_size": 50, "bos_token_id": 0, "eos_token_id": [1, 2]}
    rows = configs.make_workload(value, model)
    assert rows == configs.make_workload(value, model)
    lengths = [len(row["input_ids"]) for row in rows]
    assert min(lengths) >= 8 and max(lengths) <= 64 and len(set(lengths)) > 20
    # Fixed lengths keep the token sequence they always had.
    fixed = configs.load_scenario(
        write(tmp_path, scenario(**{"workload.context_length": 64}))
    )
    old = random.Random(fixed.workload.seed)
    allowed = list(range(3, 50))
    first = old.choices(allowed, k=64)
    assert configs.make_workload(fixed, model)[0]["input_ids"] == first


def test_cpu_list():
    assert runner.cpu_list([10, 0, 1, 2, 5, 30, 31]) == "0-2,5,10,30-31"


def test_cpu_numbers_reads_what_cpu_list_writes():
    assert runner.cpu_numbers("0-2,5,10,30-31") == [0, 1, 2, 5, 10, 30, 31]
    for text in ("5-", "a", "1,,2"):
        with pytest.raises(argparse.ArgumentTypeError):
            runner.cpu_numbers(text)


def arguments(tmp_path, catalog=None):
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text(
        json.dumps({"torch_dtype": "bfloat16", "vocab_size": 5000})
    )
    for name in ("model.safetensors", TRIE_HEAD_FILE):
        (model / name).write_text("")
    resources = configs.Resources(list(range(10)), [10], "0")
    return Namespace(model=model, catalog=catalog, resources=resources)


def make_run(tmp_path, name, output=None, catalog=False):
    found = pinned(shipped()) if catalog else shipped()
    adapter, config = configs.load_solution(path(name), found)
    output = output or tmp_path / "out"
    return runner.Run(name, adapter, found, config, tmp_path / "src", "img", output)


def test_docker_command_isolates_labels_and_mounts(tmp_path):
    run = make_run(tmp_path, VEDANTJH2, tmp_path, catalog=True)
    argv = runner.docker_command(
        arguments(tmp_path, write_catalog(tmp_path)),
        run,
        tmp_path / "c",
        "img",
        "bench-x",
        "profile",
    )
    joined = " ".join(argv)
    assert argv[:8] == [
        "docker",
        "run",
        "--rm",
        "--init",
        "--name",
        "bench-x",
        "--label",
        runner.LABEL,
    ]
    assert "--network none" in joined and "--cpuset-cpus 0-10" in joined
    assert f"src={tmp_path}/model,dst=/model-source,readonly" in joined
    assert f"src={tmp_path}/.work/model,dst=/model,readonly" in joined
    assert f"src={tmp_path}/src,dst=/sglang,readonly" in joined
    assert f"src={runner.BENCH},dst=/benchmark-tools/bench,readonly" in joined
    expected = ["--config", "/work/run_config.json", "--run", "profile", "--profile"]
    assert argv[-5:] == expected


def write_catalog(tmp_path):
    catalog = tmp_path / "items.jsonl"
    rows = [{"item_id": i, "token_ids": [1000 + i, 2000, 3000]} for i in range(256)]
    catalog.write_text("\n".join(json.dumps(row) for row in rows))
    return catalog


def catalog_scenario(tmp_path):
    """A scenario file that pins the catalog write_catalog writes."""
    data = write_catalog(tmp_path).read_bytes()
    return write(
        tmp_path, scenario(catalog={"sha256": hashlib.sha256(data).hexdigest()})
    )


def prepare(tmp_path, name):
    run = make_run(tmp_path, name, catalog=True)
    run.work.mkdir(parents=True)
    rows = configs.make_workload(run.scenario, {"vocab_size": 5000})
    info = runner.prepare(
        arguments(tmp_path, write_catalog(tmp_path)), run, {"vocab_size": 5000}, rows
    )
    return run, info


def test_prepare_writes_the_run_config_and_vedantjh2_model_view(tmp_path):
    run, info = prepare(tmp_path, VEDANTJH2)
    assert info["items"] == 256 and info["name"] == "items.jsonl"
    # The container reads the same typed configuration, as JSON.
    config = configs.RunConfig.read(run.work / "run_config.json")
    assert config.scenario == run.scenario and config.solution == run.solution
    assert config.resources.server_cpus == list(range(10))
    assert config.resources.client_cpus == [10]
    assert config.vocab_size == 5000 and config.catalog_items == 256
    assert len(read_json(run.work / "workload.json")) == 510
    assert run.adapter.catalog_source == "/model/trie_output_head.safetensors"
    view = run.work / "model"
    assert (view / "config.json").readlink().as_posix() == "/model-source/config.json"
    # The trie output head is generated, not linked to the original checkpoint.
    assert not (view / TRIE_HEAD_FILE).is_symlink()


def test_prepare_converts_the_catalog_to_each_engines_native_form(tmp_path):
    for name, source, native in (
        (LEGACY, "/work/catalog.pkl", "catalog.pkl"),
        (SID, "/work/catalog.jsonl", "catalog.jsonl"),
    ):
        directory = tmp_path / name
        directory.mkdir()
        run, _ = prepare(directory, name)
        assert run.adapter.catalog_source == source
        assert (run.work / native).is_file()


def test_vedantjh2_without_a_catalog_serves_the_checkpoint_as_is(tmp_path):
    run = make_run(tmp_path, VEDANTJH2, tmp_path)
    run.work.mkdir(parents=True)
    runner.prepare(arguments(tmp_path), run, {"vocab_size": 5000}, [])
    config = configs.RunConfig.read(run.work / "run_config.json")
    assert config.catalog_items is None
    assert not (run.work / "model").exists()
    argv = runner.docker_command(
        arguments(tmp_path), run, tmp_path / "c", "img", "bench-x", "round_1"
    )
    joined = " ".join(argv)
    assert f"src={tmp_path}/model,dst=/model,readonly" in joined
    assert "/model-source" not in joined


def fake_sources(tmp_path):
    for adapter in map(engines.get, engines.ENGINES):
        entry = tmp_path / adapter.mount.strip("/") / adapter.entry
        entry.parent.mkdir(parents=True, exist_ok=True)
        entry.write_text("")
    return tmp_path / "source", tmp_path / "sglang"


def command(tmp_path, *engine_names):
    sid_gr, sglang = fake_sources(tmp_path)
    argv = ["--scenario", str(path("qwen3_0_6b"))]
    argv += ["--model", str(arguments(tmp_path).model), "--dry-run"]
    argv += ["--output", str(tmp_path / "out")]
    argv += ["--server-cpus", "0-9", "--client-cpus", "10", "--gpu", "0"]
    for name in engine_names:
        mount = engines.get(raw(name)["engine"]).mount
        source = sid_gr if mount == "/source" else sglang
        argv += ["--solution", str(path(name)), str(source), "img"]
    return argv


def test_dry_run_prints_every_phase_and_writes_nothing(tmp_path, capsys):
    assert runner.main(command(tmp_path, SID)) == 0
    # five rounds, one profile
    assert capsys.readouterr().out.count("docker run") == 6
    assert not (tmp_path / "out").exists()


def dry_run_additions(out):
    """{solution name: what the benchmark adds}, read back from --dry-run output."""
    found = {}
    for block in out.split("# ")[1:]:
        name, _, rest = block.partition(": added")
        found[name] = yaml.safe_load(rest.split("\n", 1)[1].split("docker run")[0])
    return found


def test_dry_run_prints_what_the_benchmark_adds(tmp_path, capsys):
    argv = command(tmp_path, SID, LEGACY)
    argv[1] = str(catalog_scenario(tmp_path))
    assert runner.main(argv + ["--catalog", str(tmp_path / "items.jsonl")]) == 0
    added = dry_run_additions(capsys.readouterr().out)
    sid, legacy = added[SID], added[LEGACY]
    assert sid["env"]["GR_BEAM_WIDTH"] == "256" and "server_settings" not in sid
    assert sid["env"]["GR_CATALOG_JSONL"] == "/work/catalog.jsonl"
    settings = legacy["server_settings"]
    assert settings["beam_search_constraint_dict"] == "/work/catalog.pkl"
    assert settings["max_running_requests"] == 257 and settings["random_seed"] == 42
    # Only variables the benchmark manages, and the thread defaults: nothing from
    # this shell and nothing the solution already sets.
    for name, engine in ((SID, "sid-gr"), (LEGACY, "sglang-beam-search-new")):
        allowed = engines.get(engine).managed_env | {
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
        }
        assert added[name]["env"].keys() <= allowed


def test_cpus_and_gpu_come_from_the_command_line(tmp_path):
    argv = command(tmp_path, SID)
    assert runner.parse_args(argv).resources == configs.Resources(
        list(range(10)), [10], "0"
    )
    argv[argv.index("--client-cpus") + 1] = "9"
    with pytest.raises(ValueError, match="must not overlap"):
        runner.parse_args(argv)
    argv[argv.index("--client-cpus") + 1] = "10"
    argv[argv.index("--gpu") + 1] = "0,1"
    with pytest.raises(ValueError, match="one device index"):
        runner.parse_args(argv)


def test_several_solutions_get_their_own_directories(tmp_path, capsys):
    args = runner.parse_args(command(tmp_path, SID, LEGACY))
    assert args.scenario == path("qwen3_0_6b")
    runs = runner.build_runs(args, configs.load_scenario(args.scenario))
    assert [run.output for run in runs] == [args.output / SID, args.output / LEGACY]
    assert runner.main(command(tmp_path, SID, LEGACY)) == 0
    assert capsys.readouterr().out.count("docker run") == 12


def test_the_scenario_is_required(tmp_path, capsys):
    argv = command(tmp_path, SID)
    del argv[:2]
    with pytest.raises(SystemExit):
        runner.parse_args(argv)
    assert "--scenario" in capsys.readouterr().err


def test_engine_with_the_wrong_source_is_rejected(tmp_path):
    argv = command(tmp_path, SID)
    argv[argv.index("--solution") + 2] = str(tmp_path / "missing")
    with pytest.raises(ValueError, match="is not a recsys-examples checkout"):
        runner.main(argv)


def test_duplicate_solutions_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="distinct file names"):
        runner.main(command(tmp_path, SID, SID))


@pytest.mark.parametrize("name", [PUBLIC, INTERNAL])
def test_engines_without_catalog_support_cannot_take_a_catalog(tmp_path, name):
    argv = command(tmp_path, name) + ["--catalog", str(write_catalog(tmp_path))]
    argv[1] = str(catalog_scenario(tmp_path))
    with pytest.raises(ValueError, match="cannot constrain beams"):
        runner.main(argv)


def test_the_catalog_file_and_the_scenario_pin_go_together(tmp_path):
    catalog = ["--catalog", str(write_catalog(tmp_path))]
    with pytest.raises(ValueError, match="needs a scenario that pins it"):
        runner.main(command(tmp_path, SID) + catalog)
    argv = command(tmp_path, SID)
    argv[1] = str(catalog_scenario(tmp_path))
    with pytest.raises(ValueError, match="give its file with --catalog"):
        runner.main(argv)
    (tmp_path / "items.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="differs from the scenario's catalog.sha256"):
        runner.main(argv + catalog)


def test_reference_must_be_a_given_engine(tmp_path):
    argv = command(tmp_path, SID, PUBLIC) + ["--reference", "other"]
    with pytest.raises(ValueError, match="is not one of the --solution configs"):
        runner.main(argv)


def test_every_engine_has_an_image_recipe_and_a_build_case():
    build = (runner.ROOT / "images/build.sh").read_text()
    cases = re.findall(r"^  ([\w| -]+)\)$", build, re.M)
    built = {name.strip() for case in cases for name in case.split("|")}
    assert built == set(engines.ENGINES)
    for adapter in map(engines.get, engines.ENGINES):
        if adapter.recipe.startswith("images/"):
            assert (runner.ROOT / adapter.recipe).is_file(), adapter.name


def test_summary_is_against_public_sglang_by_default():
    assert runner.reference_name(Namespace(reference=None), [SID, PUBLIC]) == PUBLIC
    assert runner.reference_name(Namespace(reference=None), [SID, LEGACY]) == SID
    assert runner.reference_name(Namespace(reference=SID), [SID, PUBLIC]) == SID


def test_manifest_combines_identities_pins_and_hashes(tmp_path):
    args = runner.parse_args(command(tmp_path, SID))
    run = runner.build_runs(args, configs.load_scenario(args.scenario))[0]
    run.manifest = {"source": {"commit": "a" * 40}, "image": {"id": "sha256:1"}}
    before = {"tools": "abc123"}
    found = runner.manifest(args, run, before, {"id": "m"}, {"nsys": "nsys 1"})
    assert found["source"] == run.manifest["source"] and found["model"] == {"id": "m"}
    assert found["benchmark_code"] == "abc123" and found["nsys"] == "nsys 1"
    assert found["recipe"] == "images/Dockerfile.sid-gr"
    assert set(found["configs"]) == {"scenario", "solution"}


def test_a_load_profile_follows_with_several_levels_unless_steps_overlap():
    from types import SimpleNamespace

    levels = dataclasses.replace(
        shipped(), workload=dataclasses.replace(shipped().workload, concurrency=[1, 4])
    )

    def run(name, **settings):
        return SimpleNamespace(
            adapter=engines.get(name),
            solution=SimpleNamespace(server_settings=settings),
        )

    off = run("sglang", disable_overlap_schedule=True)
    assert runner.run_phases(levels, off)[-2:] == ["profile", "load_profile"]
    assert (
        runner.run_phases(shipped(), off)[-1] == "profile"
    )  # one level: nothing to load
    assert "load_profile" not in runner.run_phases(levels, run("sglang"))  # overlap on
    assert runner.run_phases(levels, run("sid-gr"))[-1] == "load_profile"
