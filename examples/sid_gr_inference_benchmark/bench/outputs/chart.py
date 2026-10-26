# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static SVG charts for report.md and summary.md, styled by
templates/chart.css; opened directly in a browser, every mark shows its numbers
on hover. The reports' tables hold the same numbers.

lines()    one small panel per measure (throughput, latency, ...) against the
           concurrency levels, one line per series, so no chart needs two y-scales
stacked()  one horizontal bar per row, split into parts, such as where a request's
           time goes at each concurrency level
"""

import math
import re
from pathlib import Path
from xml.sax.saxutils import escape

STYLE = (Path(__file__).resolve().parent / "templates" / "chart.css").read_text()
SLOTS = len(set(re.findall(r"\.l(\d+) ", STYLE)))  # series colors, in fixed order
GAP = 2  # surface gap between touching marks, px


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


def svg(width, height, title, body):
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="{escape(title)}">\n'
        f'<style><![CDATA[\n{STYLE}]]></style>\n<rect class="surface" width="{width}" height="{height}"/>\n'
        f"{body}\n</svg>\n"
    )


def legend(names, slots, left, top, width, swatch):
    """Rows of key and name; swatch "line" for lines, "box" for bars. Returns
    (svg, height)."""
    parts, x, y = [], left, top
    for name in names:
        size = 40 + 7.2 * len(name)
        if x + size > width - 16 and x > left:
            x, y = left, y + 22
        i = slots[name]
        key = (
            f'<line class="line l{i}" x1="{x}" y1="{y}" x2="{x + 22}" y2="{y}"/>'
            f'<circle class="dot d{i}" cx="{x + 11}" cy="{y}" r="4.5"/>'
            if swatch == "line"
            else f'<rect class="d{i}" x="{x + 5}" y="{y - 6}" width="12" height="12" rx="2"/>'
        )
        parts.append(
            key
            + f'<text class="legend" x="{x + 30}" y="{y + 4.5}">{escape(name)}</text>'
        )
        x += size + 18
    return "".join(parts), y - top + 22


def lines(title, subtitle, levels, panels):
    """Small multiples against the concurrency levels: panels is [(panel title,
    unit, {series: [(value, low, high) per level]})]; low and high (a 95%
    interval) may be None. A series keeps one color in every panel; a legend
    lists the series of panels that hold several, and a lone series is named by
    its panel title. None with more series than color slots."""
    slots = {}
    for _, _, series in panels:
        for name in series:
            slots.setdefault(name, len(slots))
    if len(slots) > SLOTS:
        return None
    count = len(panels)
    panel_w, gap, left = (300 if count <= 2 else 236), 64, 60
    width = left + count * panel_w + (count - 1) * gap + 24
    shared = [n for n in slots if any(n in s and len(s) > 1 for _, _, s in panels)]
    keys, key_height = legend(shared, slots, left, 78, width, "line")
    top = 78 + (key_height + 12 if shared else 0) + 26
    plot_h, height = 230, top + 230 + 56
    body = [
        f'<text class="title" x="{left}" y="30">{escape(title)}</text>',
        f'<text class="subtitle" x="{left}" y="50">{escape(subtitle)}</text>',
        keys if shared else "",
    ]
    for p, (panel_title, unit, series) in enumerate(panels):
        x0 = left + p * (panel_w + gap)
        highs = [
            point[2] if point[2] is not None else point[0]
            for points in series.values()
            for point in points
        ]
        ys = ticks(max(highs) * 1.05)
        step = panel_w / len(levels)

        def px(i, x0=x0, step=step):
            return x0 + step * (i + 0.5)

        def py(value, ys=ys):
            return top + plot_h - plot_h * value / ys[-1]

        body.append(
            f'<text class="panel-title" x="{x0}" y="{top - 12}">{escape(panel_title)}</text>'
        )
        for value in ys:
            y = py(value)
            body.append(
                f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x0 + panel_w}" y2="{y:.1f}"/>'
                f'<text class="tick" x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end">{number(value)}</text>'
            )
        body.append(
            f'<line class="axis" x1="{x0}" y1="{top + plot_h}" x2="{x0 + panel_w}" y2="{top + plot_h}"/>'
        )
        for i, level in enumerate(levels):
            body.append(
                f'<text class="tick" x="{px(i):.1f}" y="{top + plot_h + 18}" text-anchor="middle">{level}</text>'
            )
        body.append(
            f'<text class="axis-title" x="{x0 + panel_w / 2:.1f}" y="{top + plot_h + 40}" '
            'text-anchor="middle">Concurrency</text>'
        )
        marks = []
        for name, points in series.items():
            i = slots[name]
            coords = [(px(j), py(value)) for j, (value, _, _) in enumerate(points)]
            path = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
            body.append(f'<polyline class="line l{i}" points="{path}"/>')
            for j, ((value, low, high), (x, y)) in enumerate(zip(points, coords)):
                tip = f"{name} at concurrency {levels[j]}: {value:,.1f} {unit}"
                if low is not None and high is not None:
                    tip += f" (95% interval {low:,.1f} to {high:,.1f})"
                    body.append(
                        f'<line class="whisker l{i}" x1="{x:.1f}" y1="{py(low):.1f}" x2="{x:.1f}" y2="{py(high):.1f}"/>'
                    )
                marks.append(
                    f'<g><title>{escape(tip)}</title><circle class="hit" cx="{x:.1f}" cy="{y:.1f}" r="12"/>'
                    f'<circle class="dot d{i}" cx="{x:.1f}" cy="{y:.1f}" r="4.5"/></g>'
                )
        body += marks
    return svg(width, height, title, "\n".join(body))


def stacked(title, subtitle, axis, unit, parts, rows):
    """Horizontal bars, one per row: rows is [(label, [value per part])]; each
    bar is split into its parts, in order, and labeled with its total. axis
    titles the value axis; unit follows each value on hover. None with more
    parts than color slots."""
    slots = {name: i for i, name in enumerate(parts)}
    if len(parts) > SLOTS:
        return None
    width, left, right = 760, 116, 84
    keys, key_height = legend(parts, slots, left, 78, width, "box")
    top = 78 + key_height + 18
    row_h, bar = 36, 20
    plot_w = width - left - right
    height = top + row_h * len(rows) + 56
    xs = ticks(max(sum(values) for _, values in rows) * 1.02)

    def px(value):
        return left + plot_w * value / xs[-1]

    body = [
        f'<text class="title" x="{left}" y="30">{escape(title)}</text>',
        f'<text class="subtitle" x="{left}" y="50">{escape(subtitle)}</text>',
        keys,
    ]
    bottom = top + row_h * len(rows)
    for value in xs:
        x = px(value)
        body.append(
            f'<line class="grid" x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{bottom}"/>'
            f'<text class="tick" x="{x:.1f}" y="{bottom + 18}" text-anchor="middle">{number(value)}</text>'
        )
    body.append(
        f'<text class="axis-title" x="{left + plot_w}" y="{bottom + 40}" text-anchor="end">{escape(axis)}</text>'
    )
    for r, (label, values) in enumerate(rows):
        y = top + r * row_h + (row_h - bar) / 2
        total = sum(values)
        body.append(
            f'<text class="label" x="{left - 10}" y="{y + bar / 2 + 4.5:.1f}" text-anchor="end">{escape(label)}</text>'
        )
        start, last = 0.0, max(i for i, v in enumerate(values) if v > 0)
        for i, value in enumerate(values):
            if value <= 0:
                continue
            x, w = px(start), px(start + value) - px(start)
            start += value
            share = 100 * value / total
            tip = f"{label}: {parts[i]} {value:,.1f} {unit} ({share:.0f}%)"
            if i == last:  # rounded data end, square where it meets the previous part
                r_ = min(4.0, w / 2)
                shape = (
                    f'<path class="d{slots[parts[i]]}" d="M{x:.1f},{y:.1f} h{w - r_:.1f} '
                    f"a{r_:.1f},{r_:.1f} 0 0 1 {r_:.1f},{r_:.1f} v{bar - 2 * r_:.1f} "
                    f'a{r_:.1f},{r_:.1f} 0 0 1 {-r_:.1f},{r_:.1f} h{-(w - r_):.1f} z"/>'
                )
            else:
                shape = (
                    f'<rect class="d{slots[parts[i]]}" x="{x:.1f}" y="{y:.1f}" '
                    f'width="{max(w - GAP, 0.5):.1f}" height="{bar}"/>'
                )
            body.append(f"<g><title>{escape(tip)}</title>{shape}</g>")
        body.append(
            f'<text class="value" x="{px(total) + 8:.1f}" y="{y + bar / 2 + 4.5:.1f}">{total:,.1f}</text>'
        )
    return svg(width, height, title, "\n".join(body))
