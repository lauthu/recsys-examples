# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from argparse import Namespace

import pytest

import run_benchmark as runner
from bench import config as configs
from bench import engines
from bench.common import read_json

CONFIGS = ("sid_gr", "sglang", "sglang_beam_search_new", "sglang_sidtrie")
SID, PUBLIC, LEGACY, SIDTRIE = CONFIGS


def path(name):
    return runner.ROOT / "configs" / f"{name}.json"


def benchmark(**changes):
    return override(read_json(path("benchmark")), changes)


def engine_config(name=SID, **changes):
    return override(read_json(path(name)), changes)


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
    file = tmp_path / "config.json"
    file.write_text(json.dumps(value))
    return file


def test_shipped_configs_are_valid():
    shipped = configs.load_benchmark(path("benchmark"))
    for name in CONFIGS:
        engine, config = configs.load_engine(path(name), shipped["workload"])
        assert engine.name == config["engine"]


INVALID_BENCHMARK = [
    ({"extra": 1}, "exactly the keys"),
    ({"cpus.client": [9]}, "overlap"),
    ({"cpus.client": 10}, "list"),
    ({"workload.output_tokens": 1}, "output_tokens"),
    ({"gpu": 0}, "gpu"),
]


@pytest.mark.parametrize("changes, message", INVALID_BENCHMARK)
def test_invalid_benchmark_configs_fail_before_launch(tmp_path, changes, message):
    with pytest.raises(ValueError, match=message):
        configs.load_benchmark(write(tmp_path, benchmark(**changes)))


INVALID_ENGINE = [
    (SID, {"extra": 1}, "exactly the keys"),
    (SID, {"env.GR_INFERENCE_PREFILL_SDPA_ENABLE_GQA": True}, "strings"),
    (SID, {"env.GR_BEAM_WIDTH": "128"}, "runner-managed"),
    (SID, {"engine": "other"}, "engine must be one of"),
    (LEGACY, {"server_settings.enable_beam_search": False}, "enable_beam_search"),
    (LEGACY, {"server_settings.port": 30000}, "must not set"),
    (LEGACY, {"server_settings.tp_size": 2}, "tp_size"),
    (PUBLIC, {"server_settings.tp_size": 2}, "tp_size"),
    (PUBLIC, {"server_settings.cuda_graph_bs_decode": [128]}, r"must be \[256\]"),
    (LEGACY, {"server_settings.cuda_graph_bs_decode": [128]}, r"must be \[256\]"),
    (LEGACY, {"server_settings.cuda_graph_bs_prefill": [1024]}, r"must be \[2048\]"),
    (SIDTRIE, {"server_settings.enable_beam_search": True}, "beam_width API"),
    (SIDTRIE, {"env.SGLANG_BEAM_SHARED_CONTEXT_ATTENTION": "1"}, '"true"'),
    (SIDTRIE, {"trie_config.codebook_size": 0}, "codebook_size"),
]


@pytest.mark.parametrize("name, changes, message", INVALID_ENGINE)
def test_invalid_engine_configs_fail_before_launch(tmp_path, name, changes, message):
    shipped = benchmark()["workload"]
    with pytest.raises(ValueError, match=message):
        configs.load_engine(write(tmp_path, engine_config(name, **changes)), shipped)


def test_sidtrie_needs_three_output_tokens(tmp_path):
    workload = benchmark(**{"workload.output_tokens": 4})["workload"]
    with pytest.raises(ValueError, match="three SID codebooks"):
        configs.load_engine(path(SIDTRIE), workload)


def test_workload_is_seeded_and_skips_special_tokens():
    value = benchmark(**{"workload.context_length": 64})
    model = {"vocab_size": 50, "bos_token_id": 0, "eos_token_id": [1, 2]}
    rows = configs.make_workload(value, model)
    assert rows == configs.make_workload(value, model)
    assert len(rows) == 210 and rows[10]["request_id"] == "request-10"
    assert {t for row in rows for t in row["input_ids"]} <= set(range(3, 50))


def test_cpu_list():
    assert runner.cpu_list([10, 0, 1, 2, 5, 30, 31]) == "0-2,5,10,30-31"


def arguments(tmp_path, catalog=None):
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text(
        json.dumps({"torch_dtype": "bfloat16", "vocab_size": 5000})
    )
    for name in ("model.safetensors", engines.SIDTRIE_FILE):
        (model / name).write_text("")
    return Namespace(model=model, catalog=catalog)


def make_run(tmp_path, name, output=None):
    shipped = benchmark()
    engine, config = configs.load_engine(path(name), shipped["workload"])
    output = output or tmp_path / "out"
    return runner.Run(
        name, engine, {**shipped, **config}, tmp_path / "src", "img", output
    )


def test_docker_command_isolates_labels_and_mounts(tmp_path):
    run = make_run(tmp_path, SIDTRIE, tmp_path)
    argv = runner.docker_command(
        arguments(tmp_path), run, tmp_path / "c", "img", "bench-x", "profile"
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
    assert argv[-5:] == ["--plan", "/work/plan.json", "--run", "profile", "--profile"]


def write_catalog(tmp_path):
    catalog = tmp_path / "items.jsonl"
    rows = [{"item_id": i, "token_ids": [1000 + i, 2000, 3000]} for i in range(256)]
    catalog.write_text("\n".join(json.dumps(row) for row in rows))
    return catalog


def prepare(tmp_path, name):
    run = make_run(tmp_path, name)
    run.work.mkdir(parents=True)
    rows = configs.make_workload(run.config, {"vocab_size": 5000})
    info = runner.prepare(
        arguments(tmp_path, write_catalog(tmp_path)), run, {"vocab_size": 5000}, rows
    )
    return run, info


def test_prepare_writes_plan_and_sidtrie_model_view(tmp_path):
    run, info = prepare(tmp_path, SIDTRIE)
    assert info["items"] == 256 and info["name"] == "items.jsonl"
    plan = read_json(run.work / "plan.json")
    assert plan["catalog"]["server_source"] == "/model/trie_output_head.safetensors"
    assert plan["requests"] == 200 and len(read_json(run.work / "workload.json")) == 210
    view = run.work / "model"
    assert (view / "config.json").readlink().as_posix() == "/model-source/config.json"
    assert not (view / engines.SIDTRIE_FILE).is_symlink()  # generated, not the original


def test_prepare_converts_the_catalog_to_each_engines_native_form(tmp_path):
    for name, source, native in (
        (LEGACY, "/work/catalog.pkl", "catalog.pkl"),
        (SID, "/work/catalog.jsonl", "catalog.jsonl"),
    ):
        directory = tmp_path / name
        directory.mkdir()
        run, _ = prepare(directory, name)
        assert read_json(run.work / "plan.json")["catalog"]["server_source"] == source
        assert (run.work / native).is_file()


def test_sidtrie_requires_a_catalog(tmp_path):
    run = make_run(tmp_path, SIDTRIE)
    run.work.mkdir(parents=True)
    with pytest.raises(ValueError, match="needs --catalog"):
        runner.prepare(arguments(tmp_path), run, {"vocab_size": 5000}, [])


def fake_sources(tmp_path):
    for engine in engines.ENGINES.values():
        entry = tmp_path / engine.mount.strip("/") / engine.entry
        entry.parent.mkdir(parents=True, exist_ok=True)
        entry.write_text("")
    return tmp_path / "source", tmp_path / "sglang"


def command(tmp_path, *engine_names):
    sid_gr, sglang = fake_sources(tmp_path)
    argv = ["--model", str(arguments(tmp_path).model), "--dry-run"]
    argv += ["--output", str(tmp_path / "out")]
    for name in engine_names:
        source = sid_gr if name == SID else sglang
        argv += ["--engine", str(path(name)), str(source), "img"]
    return argv


def test_dry_run_prints_every_phase_and_writes_nothing(tmp_path, capsys):
    assert runner.main(command(tmp_path, SID)) == 0
    # three clean runs, one profile
    assert capsys.readouterr().out.count("docker run") == 4
    assert not (tmp_path / "out").exists()


def test_several_engines_get_their_own_directories(tmp_path, capsys):
    args = runner.parse_args(command(tmp_path, SID, LEGACY))
    runs = runner.build_runs(args, configs.load_benchmark(args.benchmark))
    assert [run.output for run in runs] == [args.output / SID, args.output / LEGACY]
    assert runner.main(command(tmp_path, SID, LEGACY)) == 0
    assert capsys.readouterr().out.count("docker run") == 8


def test_engine_with_the_wrong_source_is_rejected(tmp_path):
    argv = command(tmp_path, SID)
    argv[argv.index("--engine") + 2] = str(tmp_path / "missing")
    with pytest.raises(ValueError, match="is not a recsys-examples checkout"):
        runner.main(argv)


def test_duplicate_engine_configs_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="distinct file names"):
        runner.main(command(tmp_path, SID, SID))


def test_public_sglang_cannot_take_a_catalog(tmp_path):
    argv = command(tmp_path, PUBLIC) + ["--catalog", str(write_catalog(tmp_path))]
    with pytest.raises(ValueError, match="cannot constrain beams"):
        runner.main(argv)


def test_reference_must_be_a_given_engine(tmp_path):
    argv = command(tmp_path, SID, PUBLIC) + ["--reference", "other"]
    with pytest.raises(ValueError, match="is not one of the --engine configs"):
        runner.main(argv)


def test_every_shipped_engine_config_pins_its_source():
    for name in CONFIGS:
        pin = read_json(path(name))["source"]
        assert pin["commit"] == pin["commit"].lower() and len(pin["commit"]) == 40
    for name in ("benchmark", "benchmark_qwen3_1_7b"):
        assert configs.load_benchmark(path(name))["model"]["revision"]


def test_every_engine_names_an_image_recipe_that_exists():
    for engine in engines.ENGINES.values():
        assert (runner.ROOT / engine.dockerfile).is_file(), engine.name
