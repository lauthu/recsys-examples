# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""throughput.svg: p99 latency against requests/s, one line per engine through its
concurrency levels, in increasing order. Static SVG for comparison.md, in light or
dark colors after the viewer's color scheme; opened directly in a browser, every
point shows its numbers on hover. The comparison's table holds the same numbers.
"""

import math
from xml.sax.saxutils import escape

# Categorical slots in fixed order, (light, dark), checked with the dataviz
# validator on the surfaces below: assign by engine order, never cycle.
SERIES = (
    ("#2a78d6", "#3987e5"),
    ("#eb6834", "#d95926"),
    ("#1baf7a", "#199e70"),
    ("#eda100", "#c98500"),
    ("#e87ba4", "#d55181"),
    ("#008300", "#008300"),
    ("#6250d6", "#9085e9"),
    ("#e34948", "#e66767"),
)
# role: (light, dark)
CHROME = {
    "surface": ("#fcfcfb", "#1a1a19"),
    "ink": ("#0b0b0b", "#f0efec"),
    "ink-2": ("#52514e", "#c3c2b7"),
    "muted": ("#898781", "#898781"),
    "grid": ("#e1e0d9", "#2c2c2a"),
    "axis": ("#c3c2b7", "#383835"),
}
WIDTH, HEIGHT = 760, 480
LEFT, RIGHT, BOTTOM = 64, 136, 52  # plot margins; the right one holds end labels
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
DIRECT_LABELS = 4  # engines up to which each line also carries its name


def ticks(high, count=5):
    """Clean tick values from 0 to at least `high`."""
    raw = max(high, 1e-9) / count
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    values = [0.0]
    while values[-1] < high:
        values.append(values[-1] + step)
    return values


def number(value):
    return f"{value:,.0f}" if value >= 10 or value == 0 else f"{value:g}"


def colors(mode):
    """Color rules for one mode (0 light, 1 dark); plain values rather than CSS
    variables, which some Markdown and image renderers ignore."""
    role = {name: pair[mode] for name, pair in CHROME.items()}
    rules = [
        f".surface {{ fill: {role['surface']}; }}",
        f"text {{ fill: {role['ink']}; }}",
        f".subtitle, .axis-title {{ fill: {role['ink-2']}; }}",
        f".tick {{ fill: {role['muted']}; }}",
        f".grid {{ stroke: {role['grid']}; }}",
        f".axis {{ stroke: {role['axis']}; }}",
        f".dot {{ stroke: {role['surface']}; }}",
    ]
    rules += [
        f".l{i} {{ stroke: {pair[mode]}; }} .d{i} {{ fill: {pair[mode]}; }}"
        for i, pair in enumerate(SERIES)
    ]
    return "\n".join(rules)


def style():
    return f"""
text {{ font-family: {FONT}; }}
.title {{ font-size: 16px; font-weight: 600; }}
.subtitle, .axis-title {{ font-size: 12px; }}
.tick {{ font-size: 12px; font-variant-numeric: tabular-nums; }}
.legend, .label {{ font-size: 13px; }}
.grid, .axis {{ stroke-width: 1; }}
.line {{ fill: none; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }}
.dot {{ stroke-width: 2; }}
.hit {{ fill: transparent; }}
{colors(0)}
@media (prefers-color-scheme: dark) {{
{colors(1)}
}}"""


def legend(names, top):
    """Rows of line-key and name; returns (svg, height)."""
    parts, x, y = [], LEFT, top
    for i, name in enumerate(names):
        width = 40 + 7.2 * len(name)
        if x + width > WIDTH - 16 and x > LEFT:
            x, y = LEFT, y + 22
        parts.append(
            f'<line class="line l{i}" x1="{x}" y1="{y}" x2="{x + 22}" y2="{y}"/>'
            f'<circle class="dot d{i}" cx="{x + 11}" cy="{y}" r="4.5"/>'
            f'<text class="legend" x="{x + 30}" y="{y + 4.5}">{escape(name)}</text>'
        )
        x += width + 18
    return "".join(parts), y - top + 22


def render(curves, title, subtitle):
    """curves: {engine: [(concurrency, requests/s, p99 ms), ...]} in engine order.
    None when there are more engines than categorical slots."""
    names = list(curves)
    if not names or len(names) > len(SERIES):
        return None
    keys, key_height = legend(names, 78)
    top = 78 + key_height + 18
    plot_w, plot_h = WIDTH - LEFT - RIGHT, HEIGHT - top - BOTTOM
    xs = ticks(max(rate for points in curves.values() for _, rate, _ in points) * 1.05)
    ys = ticks(max(p99 for points in curves.values() for _, _, p99 in points) * 1.05)

    def px(rate):
        return LEFT + plot_w * rate / xs[-1]

    def py(ms):
        return top + plot_h - plot_h * ms / ys[-1]

    grid = []
    for value in ys:
        y = py(value)
        grid.append(
            f'<line class="grid" x1="{LEFT}" y1="{y:.1f}" x2="{LEFT + plot_w}" y2="{y:.1f}"/>'
        )
        grid.append(
            f'<text class="tick" x="{LEFT - 8}" y="{y + 4:.1f}" text-anchor="end">{number(value)}</text>'
        )
    for value in xs:
        x = px(value)
        grid.append(
            f'<text class="tick" x="{x:.1f}" y="{top + plot_h + 18}" text-anchor="middle">{number(value)}</text>'
        )
    grid.append(
        f'<line class="axis" x1="{LEFT}" y1="{top + plot_h}" x2="{LEFT + plot_w}" y2="{top + plot_h}"/>'
    )

    lines, dots, labels, ends = [], [], [], []
    for i, (name, points) in enumerate(curves.items()):
        coords = [(px(rate), py(p99)) for _, rate, p99 in points]
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
        lines.append(f'<polyline class="line l{i}" points="{path}"/>')
        for (k, rate, p99), (x, y) in zip(points, coords):
            tip = f"{name} · concurrency {k}: {rate:.2f} requests/s, p99 {p99:.1f} ms"
            dots.append(
                f'<g><title>{escape(tip)}</title><circle class="hit" cx="{x:.1f}" cy="{y:.1f}" r="11"/>'
                f'<circle class="dot d{i}" cx="{x:.1f}" cy="{y:.1f}" r="4.5"/></g>'
            )
        ends.append((coords[-1][1], coords[-1][0], name))
    if len(names) <= DIRECT_LABELS:
        placed = []
        for y, x, name in sorted(
            ends
        ):  # top to bottom; skip a label that would collide
            if all(abs(y - other) >= 16 for other in placed):
                placed.append(y)
                # Right of the last point, or left of it where it would leave the chart.
                left = x + 10 + 7.2 * len(name) > WIDTH - 8
                anchor = (
                    f'x="{x - 10:.1f}" text-anchor="end"'
                    if left
                    else f'x="{x + 10:.1f}"'
                )
                labels.append(
                    f'<text class="label" {anchor} y="{y + 4.5:.1f}">{escape(name)}</text>'
                )

    axis_titles = (
        f'<text class="axis-title" x="{LEFT}" y="{top - 10}">p99 latency (ms)</text>'
        f'<text class="axis-title" x="{LEFT + plot_w}" y="{HEIGHT - 12}" text-anchor="end">Requests/s</text>'
    )
    body = "\n".join(
        [
            f'<rect class="surface" width="{WIDTH}" height="{HEIGHT}"/>',
            f'<text class="title" x="{LEFT}" y="30">{escape(title)}</text>',
            f'<text class="subtitle" x="{LEFT}" y="50">{escape(subtitle)}</text>',
            keys,
            *grid,
            axis_titles,
            *lines,
            *dots,
            *labels,
        ]
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" class="viz-root" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}" role="img" aria-label="{escape(title)}">\n'
        f"<style>{style()}\n</style>\n{body}\n</svg>\n"
    )
