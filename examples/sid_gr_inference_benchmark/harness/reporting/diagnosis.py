"""Render saved diagnosis data without a GPU or profiler."""

from pathlib import Path
from string import Template

from harness.artifacts import read
from harness.config import ROOT
from harness.reporting.report import details, number, table


def render(directory):
    directory = Path(directory)
    records = read(directory / "diagnosis.json")
    run_dir = directory.parents[1]
    request = read(directory / records[0]["solution"] / "timeline/request.json")
    case = request["case"]
    measured, kernels = [], []
    for record in records:
        name, summary = record["solution"], record["summary"]
        measured.append(
            [
                name,
                summary["status"],
                summary.get("measured_requests", "—"),
                *[
                    number(summary.get(key))
                    for key in (
                        "measured_window_ms",
                        "gpu_active_union_ms",
                        "kernel_duration_sum_ms",
                    )
                ],
            ]
        )
        if summary["status"] != "success":
            kernels.append(
                table(
                    ["Candidate", "Capture issue"],
                    [[name, summary.get("reason", summary["status"])]],
                )
            )
            continue
        total = summary["kernel_duration_sum_ms"]
        kernels.append(
            details(
                f"{name}: top 10 kernels",
                table(
                    ["Kernel", "Calls", "GPU ms", "Share %"],
                    [
                        [
                            f"`{r['name']}`",
                            r["calls"],
                            number(r["gpu_ms"]),
                            number(100 * r["gpu_ms"] / total),
                        ]
                        for r in summary["kernels"][:10]
                    ],
                    right_align=(1, 2, 3),
                ),
            )
        )
    text = Template(
        (ROOT / "harness/reporting/templates/diagnosis.md").read_text()
    ).substitute(
        case_id=case["id"],
        original=f"[{run_dir.name}](../../report.md)",
        workload=table(
            [
                "Context",
                "Batch",
                "Beam",
                "Output tokens",
                "Warmups",
                "Profiled repeats",
            ],
            [
                [
                    case["context_length"],
                    case["batch_size"],
                    case["beam_width"],
                    case["generation"]["output_tokens"],
                    request["measurement"]["warmup_runs"],
                    request["measurement"]["repeats"],
                ]
            ],
            right_align=range(6),
        ),
        measured=table(
            [
                "Candidate",
                "Status",
                "Measured ranges",
                "Window ms",
                "GPU-active ms",
                "Kernel sum ms",
            ],
            measured,
            right_align=(2, 3, 4, 5),
        ),
        kernels="\n\n".join(kernels),
        artifacts="[Run archive](../../raw_artifacts.tar.gz)"
        if (run_dir / "raw_artifacts.tar.gz").is_file()
        else "[Diagnosis JSON](diagnosis.json)",
    )
    (directory / "diagnosis.md").write_text(text)
