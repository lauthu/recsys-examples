# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Renders the Markdown templates in templates/. The templates hold every
word and table of the reports; the views they render are plain numbers and
names, formatted only by the filters below. A missing value fails the render
instead of printing a blank."""

import json
import shlex
from pathlib import Path

import jinja2
import yaml

DIRECTORY = Path(__file__).resolve().parent / "templates"


def num(value, digits=1):
    """1,234.5, or – for a missing value."""
    return "–" if value is None else f"{value:,.{digits}f}"


def span(bounds, digits=1):
    """A 95% interval, low–high."""
    low, high = bounds
    return f"{low:,.{digits}f}–{high:,.{digits}f}"


def pct(share, digits=1):
    """A share as a percentage: 0.123 is 12.3%."""
    return f"{100 * share:.{digits}f}%"


def change(ratio, digits=1):
    """A relative change with its sign: 0.05 is +5.0%."""
    return f"{100 * ratio:+.{digits}f}%"


def plural(noun, n):
    """1 request; 5 requests."""
    return f"{n:,} {noun}{'' if n == 1 else 's'}"


def listing(values):
    """1; 1 and 2; 1, 2 and 4."""
    values = [
        f"{value:,}" if isinstance(value, int) else str(value) for value in values
    ]
    return (
        values[0] if len(values) == 1 else ", ".join(values[:-1]) + " and " + values[-1]
    )


def cell(text):
    """Text safe inside a Markdown table cell."""
    return str(text).replace("|", "\\|")


def code(text):
    """Inline code; backticks inside would end it."""
    return "`" + str(text).replace("`", "") + "`"


def to_yaml(value):
    return yaml.safe_dump(value, sort_keys=False).rstrip()


def to_json(value):
    return json.dumps(value, indent=2)


ENVIRONMENT = jinja2.Environment(
    loader=jinja2.FileSystemLoader(DIRECTORY),
    undefined=jinja2.StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
    autoescape=False,
)
ENVIRONMENT.filters.update(
    num=num,
    span=span,
    pct=pct,
    change=change,
    listing=listing,
    plural=plural,
    cell=cell,
    code=code,
    yaml=to_yaml,
    json=to_json,
    shell=shlex.join,
)


def render(name, **view):
    return ENVIRONMENT.get_template(name).render(**view)
