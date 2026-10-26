# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SVG sunburst of server time per request: phase > section > family.

Kernel time is split by kernel family; time with no kernel running is shown as
memcpy/memset-only or no-GPU-activity slices of the phase and section it falls in.

The shapes are drawn statically, so the file also works as an image (the report
embeds it). Opened directly in a browser, an embedded script adds hover details
and click-to-zoom: click a slice to make it the whole chart, the center or Esc
to zoom back out. The script draws from the same tree the static shapes use.
"""

import json
import math
import re
from html import escape

from bench.outputs.trace import NO_PHASE

PHASE_COLORS = ("#465da5", "#97558e", "#b87838", "#549250", "#3f8f9b", "#a5544a")
# Related families share a hue: attention teal, top-k/sort/softmax orange,
# normalization purple, data movement green, elementwise pink.
FAMILY_COLORS = {
    "attention": "#16877c",
    "attention_merge": "#5fb8ae",
    "linear": "#3778cf",
    "topk": "#dd8a2b",
    "sort": "#f0b45d",
    "softmax": "#b9651a",
    "normalization": "#9063c2",
    "qk_norm": "#b794de",
    "rotary": "#a47b4f",
    "activation": "#d45d56",
    "kv_cache": "#249cb7",
    "copy": "#8d9b35",
    "gather_scatter": "#b2bf5c",
    "reduction": "#6f7b27",
    "elementwise": "#d46594",
    "math": "#e68fb4",
    "other": "#9da6b2",
}
# Non-kernel time, keyed by the trace's busy state
NON_KERNEL = {"memory": "memcpy/memset only", "idle": "no GPU activity"}
FAMILY_COLORS.update({NON_KERNEL["memory"]: "#c2c98f", NON_KERNEL["idle"]: "#dde1e7"})
GREY = "#9da6b2"
MIDDLE, BOLD = 'text-anchor="middle"', 'font-weight="650"'
CENTER = (300, 345)
RINGS = (85, 165, 225, 270)  # inner/outer radii of the phase, section and family rings


STYLE = ".s{cursor:pointer;transition:opacity .1s}#c-hit{fill:transparent}.z #c-hit{cursor:pointer}"

SCRIPT = r"""
(function () {
  "use strict";
  var DATA = __DATA__, NS = "http://www.w3.org/2000/svg";
  var CX = 300, CY = 345, INNER = 85, OUTER = 270;
  var HINT = "Hover a slice for details · click to zoom in · click the center or press Esc to zoom out";
  var svg = document.documentElement, chart = document.getElementById("chart");
  var main = document.getElementById("c-main"), sub = document.getElementById("c-sub");
  var path = document.getElementById("c-path"), share = document.getElementById("c-share");
  var root = {n: "", v: DATA.total, c: DATA.tree}, focus = root;

  function link(node, parent, depth) {
    node.p = parent; node.d = depth;
    node.id = parent ? parent.id + "/" + parent.c.indexOf(node) : "r";
    node.c.forEach(function (child) { link(child, node, depth + 1); });
  }
  link(root, null, 0);

  function ms(ns) { return (ns / DATA.requests / 1e6).toFixed(3); }
  function pct(a, b) { return (100 * a / b).toFixed(1) + "%"; }
  function names(node) {
    var parts = [];
    for (var n = node; n && n.d > 0; n = n.p) parts.unshift(n.n);
    return parts;
  }
  function make(tag, attrs) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }
  function point(r, a) { return (CX + r * Math.cos(a)).toFixed(2) + "," + (CY + r * Math.sin(a)).toFixed(2); }
  function arc(r0, r1, a0, a1) {
    var tau = 2 * Math.PI, big = a1 - a0 > Math.PI ? 1 : 0;
    function side(r, sweep, from, to) { return "A" + r + "," + r + " 0 " + big + " " + sweep + " " + point(r, to); }
    if (a1 - a0 >= tau - 1e-9) {  // a full ring needs two half arcs
      var m = a0 + Math.PI;
      return "M" + point(r1, a0) + " A" + r1 + "," + r1 + " 0 1 1 " + point(r1, m) + " A" + r1 + "," + r1 + " 0 1 1 " + point(r1, a1) +
        " L" + point(r0, a1) + " A" + r0 + "," + r0 + " 0 1 0 " + point(r0, m) + " A" + r0 + "," + r0 + " 0 1 0 " + point(r0, a0) + " Z";
    }
    return "M" + point(r1, a0) + " " + side(r1, 1, a0, a1) + " L" + point(r0, a1) + " " + side(r0, 0, a1, a0) + " Z";
  }

  function related(a, b) { return a === b || a.indexOf(b + "/") === 0 || b.indexOf(a + "/") === 0; }

  function show(node) {
    var shown = node || focus;
    main.textContent = ms(shown.v) + " ms";
    sub.textContent = shown === root ? "server time / request" : pct(shown.v, DATA.total) + " of total";
    var label = names(shown).join(" › ");
    if (node) {
      path.textContent = label;
      share.textContent = ms(node.v) + " ms/request · " + pct(node.v, DATA.total) + " of total" +
        (node.p && node.p !== root ? " · " + pct(node.v, node.p.v) + " of " + node.p.n : "");
    } else {
      path.textContent = focus === root ? "" : "Zoomed into " + label;
      share.textContent = HINT;
    }
    var slices = chart.childNodes;
    for (var i = 0; i < slices.length; i++) {
      var id = slices[i].getAttribute("data-id");
      if (id) slices[i].setAttribute("opacity", !node || related(id, node.id) ? 1 : 0.3);
    }
  }

  function zoom(node) { focus = node; svg.setAttribute("class", node === root ? "" : "z"); draw(); show(null); }

  function draw() {
    chart.replaceChildren();
    var levels = 3 - focus.d, labels = [];
    function layout(node, start, level) {
      var angle = start;
      node.c.forEach(function (child) {
        var end = angle + 2 * Math.PI * child.v / focus.v;
        if (child.v > 0) {
          var r0 = INNER + (OUTER - INNER) * level / levels, r1 = INNER + (OUTER - INNER) * (level + 1) / levels;
          var slice = make("path", {d: arc(r0, r1, angle, end), fill: child.k, stroke: "#fff", "stroke-width": 1.2, "class": "s", "data-id": child.id});
          var tip = make("title", {});
          tip.textContent = names(child).join(" / ") + ": " + ms(child.v) + " ms/request (" + pct(child.v, DATA.total) + ")";
          slice.appendChild(tip);
          slice.addEventListener("mouseenter", function () { show(child); });
          slice.addEventListener("mouseleave", function () { show(null); });
          slice.addEventListener("click", function () { if (child.c.length) zoom(child); });
          chart.appendChild(slice);
          var radius = (r0 + r1) / 2, mid = (angle + end) / 2, label = child.d === 2 ? child.n.split(".").pop() : child.d === 1 ? child.n : "";
          var room = Math.floor(((end - angle) * radius - 12) / 7);
          if (label && room < label.length) label = room >= 6 ? label.slice(0, room - 1) + "…" : "";
          if (label) {
            var x = (CX + radius * Math.cos(mid)).toFixed(2), y = (CY + radius * Math.sin(mid)).toFixed(2);
            var deg = mid * 180 / Math.PI + 90; if (deg > 90 && deg < 270) deg += 180;
            var t = make("text", {x: x, y: y, "font-size": 11, fill: "#fff", "text-anchor": "middle", "dominant-baseline": "middle",
              "font-weight": 600, transform: "rotate(" + deg.toFixed(1) + " " + x + " " + y + ")", "pointer-events": "none"});
            t.textContent = label; labels.push(t);
          }
          if (child.c.length && child.d < 3) layout(child, angle, level + 1);
        }
        angle = end;
      });
    }
    layout(focus, -Math.PI / 2, 0);
    labels.forEach(function (t) { chart.appendChild(t); });
    var hit = make("circle", {id: "c-hit", cx: CX, cy: CY, r: INNER - 1});
    hit.addEventListener("click", function () { if (focus.p) zoom(focus.p); });
    chart.appendChild(hit);
  }

  document.addEventListener("keydown", function (e) { if (e.key === "Escape") zoom(root); });
  draw(); show(null);
})();
"""


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


def render(work, gaps, requests, title, subtitle):
    """Return an SVG document; durations are means over the profiled requests.

    work rows are [phase, section, kernel family, ns]; gaps rows are
    [phase, section, "memory" | "idle", ns] from trace.analyze.
    """
    work = work + [[p, s, NON_KERNEL[state], ns] for p, s, state, ns in gaps]
    phases = [phase for phase in build(work) if phase[1] > 0]
    total = sum(ns for _, ns, _ in phases)
    if total <= 0 or requests < 1:
        raise ValueError("sunburst needs positive time and at least one request")

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

    colors, start, tree = {}, -math.pi / 2, []
    for index, (phase, phase_ns, sections) in enumerate(phases):
        colors[phase] = PHASE_COLORS[index % len(PHASE_COLORS)]
        colors[phase] = GREY if phase in ("unknown", NO_PHASE) else colors[phase]
        add(0, start, phase_ns, colors[phase], [phase], phase)
        phase_node = {"n": phase, "v": phase_ns, "k": colors[phase], "c": []}
        tree.append(phase_node)
        for rank, (section, section_ns, families) in enumerate(sections):
            shade = tint(colors[phase], 0.2 + 0.12 * (rank % 4))
            label = section.rsplit(".", 1)[-1]
            end = add(1, start, section_ns, shade, [phase, section], label)
            section_node = {"n": section, "v": section_ns, "k": shade, "c": []}
            phase_node["c"].append(section_node)
            for family, ns in families:
                color = FAMILY_COLORS.get(family, GREY)
                start = add(2, start, ns, color, [phase, section, family])
                section_node["c"].append({"n": family, "v": ns, "k": color, "c": []})
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
    heading = "Kernel family or non-kernel time (outer ring)"
    legend.append(text(600, y, heading, 12, muted, 'font-weight="600"'))
    present = [f for f in FAMILY_COLORS if any(r[2] == f and r[3] > 0 for r in work)]
    for index, family in enumerate(present):
        x, row_y = 600 + (index % 2) * 160, y + 26 + (index // 2) * 22
        legend += [
            swatch(x, row_y, FAMILY_COLORS[family]),
            text(x + 18, row_y, family, 12),
        ]
    height = max(700, y + 26 + (len(present) + 1) // 2 * 22 + 16)
    cx, cy = CENTER
    svg = f'xmlns="http://www.w3.org/2000/svg" width="960" height="{height}" '
    svg += f'viewBox="0 0 960 {height}" font-family="Segoe UI, Helvetica, Arial, sans-serif"'
    main = f'id="c-main" {MIDDLE} {BOLD}'
    center = [text(cx, cy - 4, f"{ms(total):.3f} ms", 22, extra=main)]
    center.append(
        text(cx, cy + 18, "server time / request", 11, muted, f'id="c-sub" {MIDDLE}')
    )
    detail = [text(cx, 655, "", 13, extra=f'id="c-path" {MIDDLE} {BOLD}')]
    detail.append(text(cx, 677, "", 12, muted, f'id="c-share" {MIDDLE}'))
    header = [
        f"<title>{escape(title)}</title>",
        '<rect width="100%" height="100%" fill="#fff"/>',
    ]
    header += [text(24, 36, title, 20, extra=BOLD), text(24, 60, subtitle, 13, muted)]
    data = {"total": total, "requests": requests, "tree": tree}
    data = json.dumps(data).replace("<", "\\u003c").replace("]]>", "]]\\u003e")
    script = f"<script><![CDATA[{SCRIPT.replace('__DATA__', data)}]]></script>"
    chart = ['<g id="chart">', *shapes, *labels, "</g>"]
    body = header + chart + center + detail + legend
    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f"<svg {svg}>",
            f"<style>{STYLE}</style>",
            *body,
            script,
            "</svg>",
            "",
        ]
    )
