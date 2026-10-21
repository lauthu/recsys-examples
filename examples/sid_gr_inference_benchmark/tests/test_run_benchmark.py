# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import re
from argparse import Namespace

import pytest
import run_benchmark as runner
import yaml
from bench import config as configs
from bench import engines
from bench.common import read_json

CONFIGS = ("sid_gr", "sglang", "sglang_beam_search_new", "vedantjh2_sglang")
SID, PUBLIC, LEGACY, VEDANTJH2 = CONFIGS
SCENARIOS = ("qwen3_0_6b", "qwen3_1_7b")


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
    for name in CONFIGS:
        engine, found = configs.load_solution(path(name), shipped().workload)
        assert isinstance(found, configs.Solution) and engine.name == found.engine
        assert isinstance(found.source, configs.Source)
    trie = configs.load_solution(path(VEDANTJH2), shipped().workload)[1].trie_config
    assert trie == configs.TrieConfig(token_start=1000, codebook_size=1000)


INVALID_SCENARIO = [
    ({"extra": 1}, r"unknown keys \['extra'\]"),
    ({"workload.extra": 1}, r"scenario config.yaml.workload has unknown keys"),
    ({"cpus.client": [9]}, "overlap"),
    ({"cpus.client": 10}, "cpus.client must be a list"),
    ({"cpus.client": [True]}, r"cpus.client\[0\] must be int, not bool"),
    ({"workload.output_tokens": 1}, "workload.output_tokens must be >= 2"),
    ({"workload.beam_width": 1.5}, "workload.beam_width must be int, not float"),
    ({"runs.clean": "3"}, "runs.clean must be int, not str"),
    ({"gpu": 0}, "gpu must be str, not int"),
    ({"gpu": "0,1"}, "one device index"),
    ({"model.revision": "main"}, "40-character Hub commit"),
    ({"model.sha256": {"model.safetensors": "0" * 64}}, "including config.json"),
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
    file.write_text(path("qwen3_0_6b").read_text() + '\ngpu: "1"\n')
    with pytest.raises(ValueError, match=r"duplicate keys \['gpu'\]"):
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
    (PUBLIC, {"trie_config": {"token_start": 0, "codebook_size": 1}}, "sections"),
    (LEGACY, {"server_settings.enable_beam_search": False}, "enable_beam_search"),
    (LEGACY, {"server_settings.port": 30000}, "must not set"),
    (LEGACY, {"server_settings.tp_size": 2}, "tp_size"),
    (LEGACY, {"server_settings.disable_overlap_schedule": False}, "overlap"),
    (PUBLIC, {"server_settings.tp_size": 2}, "tp_size"),
    (PUBLIC, {"server_settings.cuda_graph_bs_decode": [128]}, r"must be \[256\]"),
    (LEGACY, {"server_settings.cuda_graph_bs_decode": [128]}, r"must be \[256\]"),
    (LEGACY, {"server_settings.cuda_graph_bs_prefill": [1024]}, r"must be \[2048\]"),
    (VEDANTJH2, {"server_settings.enable_beam_search": True}, "beam_width API"),
    (VEDANTJH2, {"env.SGLANG_BEAM_SHARED_CONTEXT_ATTENTION": "1"}, '"true"'),
    (VEDANTJH2, {"trie_config.codebook_size": 0}, "codebook_size must be >= 1"),
    (VEDANTJH2, {"trie_config.depth": 3}, "trie_config has unknown keys"),
]


@pytest.mark.parametrize("name, changes, message", INVALID_SOLUTION)
def test_invalid_solutions_fail_before_launch(tmp_path, name, changes, message):
    with pytest.raises(ValueError, match=message):
        configs.load_solution(
            write(tmp_path, solution(name, **changes)), shipped().workload
        )


@pytest.mark.parametrize("name", [PUBLIC, VEDANTJH2])
def test_native_beam_search_may_use_the_overlap_scheduler(tmp_path, name):
    value = solution(name, **{"server_settings.disable_overlap_schedule": False})
    found = configs.load_solution(write(tmp_path, value), shipped().workload)[1]
    assert found.server_settings["disable_overlap_schedule"] is False


def test_sglang_solutions_need_server_settings(tmp_path):
    value = solution(PUBLIC)
    del value["server_settings"]
    with pytest.raises(ValueError, match=r"sections \['server_settings'\]"):
        configs.load_solution(write(tmp_path, value), shipped().workload)


def test_vedantjh2_needs_three_output_tokens(tmp_path):
    workload = configs.Workload(
        context_length=2048, beam_width=256, output_tokens=4, seed=0
    )
    with pytest.raises(ValueError, match="three SID codebooks"):
        configs.load_solution(path(VEDANTJH2), workload)


def test_workload_is_seeded_and_skips_special_tokens(tmp_path):
    value = configs.load_scenario(
        write(tmp_path, scenario(**{"workload.context_length": 64}))
    )
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
    for name in ("model.safetensors", engines.TRIE_HEAD_FILE):
        (model / name).write_text("")
    return Namespace(model=model, catalog=catalog)


def make_run(tmp_path, name, output=None):
    found = shipped()
    engine, config = configs.load_solution(path(name), found.workload)
    output = output or tmp_path / "out"
    return runner.Run(name, engine, found, config, tmp_path / "src", "img", output)


def test_docker_command_isolates_labels_and_mounts(tmp_path):
    run = make_run(tmp_path, VEDANTJH2, tmp_path)
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
    rows = configs.make_workload(run.scenario, {"vocab_size": 5000})
    info = runner.prepare(
        arguments(tmp_path, write_catalog(tmp_path)), run, {"vocab_size": 5000}, rows
    )
    return run, info


def test_prepare_writes_plan_and_vedantjh2_model_view(tmp_path):
    run, info = prepare(tmp_path, VEDANTJH2)
    assert info["items"] == 256 and info["name"] == "items.jsonl"
    plan = read_json(run.work / "plan.json")
    assert plan["catalog"]["server_source"] == "/model/trie_output_head.safetensors"
    assert plan["requests"] == 200 and len(read_json(run.work / "workload.json")) == 210
    # The container gets plain JSON generated from the typed configs.
    assert plan["workload"] == raw("qwen3_0_6b")["workload"]
    assert plan["server_settings"] == raw(VEDANTJH2)["server_settings"]
    assert plan["server_cpus"] == list(range(10)) and plan["client_cpus"] == [10]
    view = run.work / "model"
    assert (view / "config.json").readlink().as_posix() == "/model-source/config.json"
    # The trie output head is generated, not linked to the original checkpoint.
    assert not (view / engines.TRIE_HEAD_FILE).is_symlink()


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


def test_vedantjh2_requires_a_catalog(tmp_path):
    run = make_run(tmp_path, VEDANTJH2)
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
        argv += ["--solution", str(path(name)), str(source), "img"]
    return argv


def test_dry_run_prints_every_phase_and_writes_nothing(tmp_path, capsys):
    assert runner.main(command(tmp_path, SID)) == 0
    # three clean runs, one profile
    assert capsys.readouterr().out.count("docker run") == 4
    assert not (tmp_path / "out").exists()


def test_several_solutions_get_their_own_directories(tmp_path, capsys):
    args = runner.parse_args(command(tmp_path, SID, LEGACY))
    assert args.scenario == path("qwen3_0_6b")  # the default scenario
    runs = runner.build_runs(args, configs.load_scenario(args.scenario))
    assert [run.output for run in runs] == [args.output / SID, args.output / LEGACY]
    assert runner.main(command(tmp_path, SID, LEGACY)) == 0
    assert capsys.readouterr().out.count("docker run") == 8


def test_engine_with_the_wrong_source_is_rejected(tmp_path):
    argv = command(tmp_path, SID)
    argv[argv.index("--solution") + 2] = str(tmp_path / "missing")
    with pytest.raises(ValueError, match="is not a recsys-examples checkout"):
        runner.main(argv)


def test_duplicate_solutions_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="distinct file names"):
        runner.main(command(tmp_path, SID, SID))


def test_public_sglang_cannot_take_a_catalog(tmp_path):
    argv = command(tmp_path, PUBLIC) + ["--catalog", str(write_catalog(tmp_path))]
    with pytest.raises(ValueError, match="cannot constrain beams"):
        runner.main(argv)


def test_reference_must_be_a_given_engine(tmp_path):
    argv = command(tmp_path, SID, PUBLIC) + ["--reference", "other"]
    with pytest.raises(ValueError, match="is not one of the --solution configs"):
        runner.main(argv)


def test_every_engine_has_an_image_recipe_and_a_build_case():
    build = (runner.ROOT / "images/build.sh").read_text()
    cases = re.findall(r"^  ([\w| -]+)\)$", build, re.M)
    built = {name.strip() for case in cases for name in case.split("|")}
    assert built == set(engines.ENGINES)
    for engine in engines.ENGINES.values():
        if engine.recipe.startswith("images/"):
            assert (runner.ROOT / engine.recipe).is_file(), engine.name


def finished_run(tmp_path, name):
    from tests.test_compare import with_outputs

    run = runner.Run(name, None, None, None, tmp_path, "id", tmp_path / name)
    return run, with_outputs("sglang" if name == PUBLIC else "sid-gr", [[1, 2, 3]], 9.0)


def test_comparison_is_written_against_public_sglang_by_default(tmp_path):
    args = Namespace(output=tmp_path, reference=None)
    pairs = [finished_run(tmp_path, SID), finished_run(tmp_path, PUBLIC)]
    runner.write_comparison(args, [p[0] for p in pairs], [p[1] for p in pairs])
    assert "Output agreement with sglang" in (tmp_path / "comparison.md").read_text()


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
