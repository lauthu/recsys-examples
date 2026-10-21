# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static SVG sunburst of GPU kernel time per request: phase > section > family."""

import math
import re
from html import escape

PHASE_COLORS = ("#465da5", "#97558e", "#b87838", "#549250", "#3f8f9b", "#a5544a")
FAMILY_COLORS = {
    "attention": "#16877c",
    "linear": "#3778cf",
    "selection": "#dd8a2b",
    "normalization": "#9063c2",
    "rotary": "#a47b4f",
    "activation": "#d45d56",
    "kv_cache": "#249cb7",
    "memory": "#8d9b35",
    "elementwise": "#d46594",
    "other": "#9da6b2",
}
GREY = "#9da6b2"
MIDDLE, BOLD = 'text-anchor="middle"', 'font-weight="650"'
CENTER = (300, 345)
RINGS = (85, 165, 225, 270)  # inner/outer radii of the phase, section and family rings


def phase_key(name):
    """prefill, decode_1, decode_2, ..., other phases, unknown last."""
    decode = re.fullmatch(r"decode_(\d+)", name)
    if name == "prefill":
        return (0, 0, name)
    if decode:
        return (1, int(decode[1]), name)
    return (3 if name == "unknown" else 2, 0, name)


def tint(color, amount):
    """Blend #rrggbb toward white by amount in [0, 1]."""
    channels = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    return "#" + "".join(f"{round(c + (255 - c) * amount):02x}" for c in channels)


def build(work):
    """[[phase, section, family, ns], ...] -> [(phase, ns, [(section, ns, [(family, ns)])])]."""
    tree = {}
    for phase, section, family, ns in work:
        families = tree.setdefault(phase, {}).setdefault(section, {})
        families[family] = families.get(family, 0) + ns
    result = []
    for phase in sorted(tree, key=phase_key):
        sections = [
            (
                name,
                sum(families.values()),
                sorted(families.items(), key=lambda x: -x[1]),
            )
            for name, families in tree[phase].items()
        ]
        sections.sort(key=lambda x: -x[1])
        result.append((phase, sum(ns for _, ns, _ in sections), sections))
    return result


def arc(r0, r1, a0, a1):
    def point(r, a):
        return f"{CENTER[0] + r * math.cos(a):.2f},{CENTER[1] + r * math.sin(a):.2f}"

    if a1 - a0 >= math.tau - 1e-9:  # a full ring needs two half arcs
        mid = a0 + math.pi
        return (
            f"M{point(r1, a0)} A{r1},{r1} 0 1 1 {point(r1, mid)} A{r1},{r1} 0 1 1 {point(r1, a1)} "
            f"L{point(r0, a1)} A{r0},{r0} 0 1 0 {point(r0, mid)} A{r0},{r0} 0 1 0 {point(r0, a0)} Z"
        )
    large = int(a1 - a0 > math.pi)
    return (
        f"M{point(r1, a0)} A{r1},{r1} 0 {large} 1 {point(r1, a1)} "
        f"L{point(r0, a1)} A{r0},{r0} 0 {large} 0 {point(r0, a0)} Z"
    )


def text(x, y, value, size=13, fill="#1d2b3a", extra=""):
    attributes = f'x="{x}" y="{y}" font-size="{size}" fill="{fill}" {extra}'
    return f"<text {attributes}>{escape(str(value))}</text>"


def swatch(x, y, color):
    return f'<rect x="{x}" y="{y - 10}" width="11" height="11" rx="2" fill="{color}"/>'


def render(work, requests, title, subtitle):
    """Return an SVG document; durations are means over the profiled requests."""
    phases = [phase for phase in build(work) if phase[1] > 0]
    total = sum(ns for _, ns, _ in phases)
    if total <= 0 or requests < 1:
        raise ValueError("sunburst needs positive kernel time and at least one request")

    def ms(ns):
        return ns / requests / 1e6

    shapes, labels = [], []

    def add(ring, a0, ns, color, path, label=None):
        """Draw a slice of ns starting at angle a0; return its end angle."""
        a1 = a0 + math.tau * ns / total
        if ns <= 0:
            return a1
        tip = f"{' / '.join(path)}: {ms(ns):.3f} ms/request ({100 * ns / total:.1f}%)"
        d = arc(RINGS[ring], RINGS[ring + 1], a0, a1)
        shapes.append(f'<path d="{d}" fill="{color}" stroke="#fff" stroke-width="1.2">')
        shapes[-1] += f"<title>{escape(tip)}</title></path>"
        radius, middle = (RINGS[ring] + RINGS[ring + 1]) / 2, (a0 + a1) / 2
        room = int(((a1 - a0) * radius - 12) / 7)  # characters that fit along the arc
        if label and room < len(label):
            label = label[: room - 1] + "…" if room >= 6 else None
        if label:
            x = f"{CENTER[0] + radius * math.cos(middle):.2f}"
            y = f"{CENTER[1] + radius * math.sin(middle):.2f}"
            angle = math.degrees(middle) + 90
            angle += 180 if 90 < angle < 270 else 0
            extra = f'{MIDDLE} dominant-baseline="middle" font-weight="600" '
            extra += f'transform="rotate({angle:.1f} {x} {y})"'
            labels.append(text(x, y, label, 11, "#fff", extra))
        return a1

    colors, start = {}, -math.pi / 2
    for index, (phase, phase_ns, sections) in enumerate(phases):
        colors[phase] = PHASE_COLORS[index % len(PHASE_COLORS)]
        colors[phase] = GREY if phase == "unknown" else colors[phase]
        add(0, start, phase_ns, colors[phase], [phase], phase)
        for rank, (section, section_ns, families) in enumerate(sections):
            shade = tint(colors[phase], 0.2 + 0.12 * (rank % 4))
            label = section.rsplit(".", 1)[-1]
            end = add(1, start, section_ns, shade, [phase, section], label)
            for family, ns in families:
                color = FAMILY_COLORS.get(family, GREY)
                start = add(2, start, ns, color, [phase, section, family])
            start = end

    muted = "#5d6b7c"
    legend = [text(600, 100, "Phase (inner ring)", 12, muted, 'font-weight="600"')]
    legend.append(text(850, 100, "ms/request", 12, muted, 'text-anchor="end"'))
    legend.append(text(920, 100, "share", 12, muted, 'text-anchor="end"'))
    y = 126
    for phase, phase_ns, _ in phases:
        share = f"{100 * phase_ns / total:.1f}%"
        legend += [swatch(600, y, colors[phase]), text(618, y, phase)]
        legend.append(text(850, y, f"{ms(phase_ns):.3f}", extra='text-anchor="end"'))
        legend.append(text(920, y, share, extra='text-anchor="end"'))
        y += 24
    y += 22
    heading = "Kernel family (outer ring)"
    legend.append(text(600, y, heading, 12, muted, 'font-weight="600"'))
    present = [f for f in FAMILY_COLORS if any(r[2] == f and r[3] > 0 for r in work)]
    for index, family in enumerate(present):
        x, row_y = 600 + (index % 2) * 160, y + 26 + (index // 2) * 22
        legend += [
            swatch(x, row_y, FAMILY_COLORS[family]),
            text(x + 18, row_y, family, 12),
        ]
    height = max(640, y + 26 + (len(present) + 1) // 2 * 22 + 16)
    cx, cy = CENTER
    svg = f'xmlns="http://www.w3.org/2000/svg" width="960" height="{height}" '
    svg += f'viewBox="0 0 960 {height}" font-family="Segoe UI, Helvetica, Arial, sans-serif"'
    center = [text(cx, cy - 4, f"{ms(total):.3f} ms", 22, extra=f"{MIDDLE} {BOLD}")]
    center.append(text(cx, cy + 18, "GPU kernel time / request", 11, muted, MIDDLE))
    header = [
        f"<title>{escape(title)}</title>",
        '<rect width="100%" height="100%" fill="#fff"/>',
    ]
    header += [text(24, 36, title, 20, extra=BOLD), text(24, 60, subtitle, 13, muted)]
    body = header + shapes + labels + center + legend
    return "\n".join(
        ['<?xml version="1.0" encoding="UTF-8"?>', f"<svg {svg}>", *body, "</svg>", ""]
    )
