#!/usr/bin/env python
"""Assemble hr_rag_colab.ipynb from parts/*.py (percent format) + embedded data.

Usage:  python tools/build_notebook.py [--out PATH]
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PARTS = ROOT / "parts"
DEFAULT_OUT = ROOT / "hr_rag_colab.ipynb"

MARK = re.compile(r"^# %%(\s*\[markdown\])?\s*$")


def _load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _lit(name: str, obj) -> str:
    s = json.dumps(obj, ensure_ascii=False, indent=1)
    if "'''" in s:
        raise ValueError(f"embedded data '{name}' contains triple single quotes")
    return f"{name} = json.loads(r'''{s}''')"


def embedded_source() -> str:
    spec = {p.stem: _load_yaml(p) for p in sorted((ROOT / "spec" / "companies").glob("*.yaml"))}
    seeds = json.loads((ROOT / "spec" / "seeds" / "names.json").read_text(encoding="utf-8"))
    policies = {f"{p.parent.name}/{p.name}": p.read_text(encoding="utf-8")
                for p in sorted((ROOT / "data_src" / "policies").glob("*/*.md"))}
    ev_r = _load_yaml(ROOT / "data_src" / "eval" / "retrieval_qa.yaml")
    ev_g = _load_yaml(ROOT / "data_src" / "eval" / "guardrail_cases.yaml")
    return "\n".join([
        "# %% [markdown]",
        "# ## 0) ข้อมูลที่ฝังมากับ Notebook (สร้างอัตโนมัติ)",
        "# - `_SPEC` สเปกของ 3 บริษัท (ตัวเลขนโยบายชุดเดียวกันที่ใช้สร้างทั้งฐานข้อมูลและเอกสาร)",
        "# - `POLICY_FILES` ไฟล์นโยบาย `.md` ทั้งหมด, `_SEEDS` คลังชื่อ/ข้อความสำหรับสุ่ม",
        "# - `_EVAL_RETRIEVAL`, `_EVAL_GUARDRAIL` ชุดทดสอบ",
        "",
        "# %%",
        "import json",
        _lit("_SPEC", spec),
        "",
        "# %%",
        "import json",
        _lit("_SEEDS", seeds),
        "",
        "# %%",
        "import json",
        _lit("POLICY_FILES", policies),
        "",
        "# %%",
        "import json",
        _lit("_EVAL_RETRIEVAL", ev_r),
        _lit("_EVAL_GUARDRAIL", ev_g),
        "print('embedded:', len(_SPEC), 'companies,', len(POLICY_FILES), 'policy files,',",
        "      len(_EVAL_RETRIEVAL), 'retrieval QA,', len(_EVAL_GUARDRAIL), 'guardrail cases')",
        "",
    ])


def parse_percent(text: str, origin: str) -> list[dict]:
    cells: list[dict] = []
    cur = None
    for line in text.splitlines():
        m = MARK.match(line)
        if m:
            if cur:
                cells.append(cur)
            cur = {"kind": "markdown" if m.group(1) else "code", "lines": [], "origin": origin}
            continue
        if cur is None:
            if line.strip():
                raise ValueError(f"{origin}: content before the first '# %%' marker: {line[:40]!r}")
            continue
        cur["lines"].append(line)
    if cur:
        cells.append(cur)
    out = []
    for c in cells:
        lines = c["lines"]
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        if not lines:
            continue
        if c["kind"] == "markdown":
            lines = [re.sub(r"^# ?", "", l) for l in lines]
        c["source"] = "\n".join(lines)
        del c["lines"]
        out.append(c)
    return out


def code_for_exec(source: str) -> str:
    """Strip IPython-only lines (!pip, %magic) so the cell can be parsed by plain Python."""
    return "\n".join("pass" if re.match(r"\s*[!%]", l) else l for l in source.splitlines())


def part_number(path: Path) -> int:
    m = re.match(r"(\d+)_", path.name)
    return int(m.group(1)) if m else 999


def collect_cells() -> list[dict]:
    cells = parse_percent(embedded_source(), "00_embedded_data")
    for p in sorted(PARTS.glob("*.py"), key=lambda p: (part_number(p), p.name)):
        cells += parse_percent(p.read_text(encoding="utf-8"), p.name)
    return cells


def check_syntax(cells: list[dict]) -> list[str]:
    errs = []
    for i, c in enumerate(cells):
        if c["kind"] != "code":
            continue
        try:
            ast.parse(code_for_exec(c["source"]))
        except SyntaxError as e:
            errs.append(f"{c['origin']} (cell {i}): {e.msg} at line {e.lineno}")
    return errs


def to_notebook(cells: list[dict]) -> dict:
    nb_cells = []
    for c in cells:
        src = c["source"].split("\n")
        src = [l + "\n" for l in src[:-1]] + [src[-1]]
        base = {"cell_type": c["kind"], "metadata": {}, "source": src}
        if c["kind"] == "code":
            base.update(execution_count=None, outputs=[])
        nb_cells.append(base)
    return {
        "cells": nb_cells,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4"},
            "accelerator": "GPU",
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    cells = collect_cells()
    errs = check_syntax(cells)
    if errs:
        print("SYNTAX ERRORS:")
        for e in errs:
            print("  -", e)
        return 1
    nb = to_notebook(cells)
    Path(args.out).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    n_code = sum(c["kind"] == "code" for c in cells)
    print(f"OK: {args.out}  ({len(cells)} cells: {n_code} code, {len(cells) - n_code} markdown)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
