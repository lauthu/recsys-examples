# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from argparse import Namespace

import pytest
import run_benchmark as runner
from bench.common import read_json

CONFIGS = ("benchmark.json", "benchmark_sglang.json", "benchmark_sglang_sidtrie.json")
SID, LEGACY, SIDTRIE = CONFIGS


def config(name=SID, **changes):
    """A shipped config with dotted-path overrides such as {"workload.seed": 1}."""
    value = read_json(runner.TOOLS / name)
    for path, item in changes.items():
        *parents, key = path.split(".")
        target = value
        for parent in parents:
            target = target[parent]
        target[key] = item
    return value


@pytest.mark.parametrize("name", CONFIGS)
def test_shipped_configs_are_valid(name):
    assert runner.load_config(runner.TOOLS / name)["engine"]


INVALID = [
    (SID, {"extra": 1}, "exactly the keys"),
    (SID, {"env.GR_INFERENCE_PREFILL_SDPA_ENABLE_GQA": True}, "strings"),
    (SID, {"env.GR_BEAM_WIDTH": "128"}, "runner-managed"),
    (SID, {"cpus.client": [9]}, "overlap"),
    (SID, {"cpus.client": 10}, "list"),
    (SID, {"workload.output_tokens": 1}, "output_tokens"),
    (SID, {"gpu": 0}, "gpu"),
    (LEGACY, {"server_settings.enable_beam_search": False}, "enable_beam_search"),
    (LEGACY, {"server_settings.port": 30000}, "must not set"),
    (LEGACY, {"server_settings.tp_size": 2}, "tp_size"),
    (SIDTRIE, {"workload.output_tokens": 4}, "three SID codebooks"),
    (SIDTRIE, {"server_settings.enable_beam_search": True}, "beam_width API"),
    (SIDTRIE, {"env.SGLANG_BEAM_SHARED_CONTEXT_ATTENTION": "1"}, '"true"'),
]


@pytest.mark.parametrize("name, changes, message", INVALID)
def test_invalid_configs_fail_before_launch(tmp_path, name, changes, message):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config(name, **changes)))
    with pytest.raises(ValueError, match=message):
        runner.load_config(path)


def test_workload_is_seeded_and_skips_special_tokens():
    value = config(**{"workload.context_length": 64})
    model = {"vocab_size": 50, "bos_token_id": 0, "eos_token_id": [1, 2]}
    rows = runner.make_workload(value, model)
    assert rows == runner.make_workload(value, model)
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
    for name in ("model.safetensors", runner.SIDTRIE_FILE):
        (model / name).write_text("")
    return Namespace(source=tmp_path / "src", model=model, catalog=catalog)


def test_docker_command_isolates_labels_and_mounts(tmp_path):
    argv = runner.docker_command(
        arguments(tmp_path),
        config(SIDTRIE),
        tmp_path / "w",
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
    assert f"src={tmp_path}/w/model,dst=/model,readonly" in joined
    assert f"src={tmp_path}/src,dst=/sglang,readonly" in joined
    assert argv[-5:] == ["--plan", "/work/plan.json", "--run", "profile", "--profile"]


def test_prepare_writes_plan_and_sidtrie_model_view(tmp_path):
    catalog = tmp_path / "items.jsonl"
    rows = [{"item_id": i, "token_ids": [1000 + i, 2000, 3000]} for i in range(256)]
    catalog.write_text("\n".join(json.dumps(row) for row in rows))
    work = tmp_path / "work"
    work.mkdir()
    info = runner.prepare(
        arguments(tmp_path, catalog), config(SIDTRIE), {"vocab_size": 5000}, work
    )
    assert info["items"] == 256 and info["name"] == "items.jsonl"
    plan = read_json(work / "plan.json")
    assert plan["catalog"]["server_source"] == "/model/trie_output_head.safetensors"
    assert plan["requests"] == 200 and len(read_json(work / "workload.json")) == 210
    view = work / "model"
    assert (view / "config.json").readlink().as_posix() == "/model-source/config.json"
    assert not (view / runner.SIDTRIE_FILE).is_symlink()  # generated, not the original


def test_dry_run_writes_nothing(tmp_path, capsys):
    model = arguments(tmp_path).model
    (tmp_path / "src/examples/sid-gr-inference/scripts").mkdir(parents=True)
    (
        tmp_path / "src/examples/sid-gr-inference/scripts/serve_qwen3_gr_http.sh"
    ).write_text("")
    argv = [
        "--source",
        str(tmp_path / "src"),
        "--model",
        str(model),
        "--image",
        "img",
        "--output",
        str(tmp_path / "out"),
        "--dry-run",
    ]
    assert runner.main(argv) == 0
    assert (
        capsys.readouterr().out.count("docker run") == 4
    )  # three clean runs, one profile
    assert not (tmp_path / "out").exists()
