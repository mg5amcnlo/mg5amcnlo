#!/usr/bin/env python3
"""Check that the Doxygen comments follow the documentation convention.

Read-only. Parses ``docs/build/doxygenxml/*.xml`` (run ``doxygen`` from ``docs/``
first) and reports, per class, what is missing relative to the convention in
``docs/CONTRIBUTING-docs.md``:

phasespace/ classes:

* non-empty one-line brief and a detailed description
* ``Mapping`` subclasses: **Inputs**, **Conditions**, **Outputs**, the weight
  sentence and a **References** list
* ``FunctionGenerator`` subclasses: **Arguments**, **Returns**, **References**
* every public method documented
* every public-constructor parameter documented
* no ``\\rst`` markers (they do not survive into the Python docstrings)
* en-dash (not hyphen) used as the bullet separator
* no obvious British spellings
* every arXiv id cited with a single author/title string

compgraphs/ classes (the small in-scope set only, see ``COMPGRAPHS_CLASSES``):

* non-empty one-line brief and a detailed description (nothing else: these value
  types have many trivial overloaded constructors)

instruction_set.yaml:

* every instruction has a non-empty ``desc``
* every named input and output has a non-empty ``desc``

Exits non-zero if any in-scope class has an unmet requirement. Until the whole
directory is documented this is expected; the printed list is the to-do list.
"""

from __future__ import annotations

import glob
import os
import re
import sys
import xml.etree.ElementTree as ET

XML_DIR = os.path.join(os.path.dirname(__file__), "build", "doxygenxml")
YAML_PATH = os.path.join(
    os.path.dirname(__file__), "..", "madspace", "instruction_set.yaml"
)

# The only compgraphs/ types that carry API docs; every *Instruction subclass and
# the optimizer internals are deliberately left out.
COMPGRAPHS_CLASSES = {
    "BatchSize",
    "Type",
    "Value",
    "InstructionCall",
    "Function",
    "FunctionBuilder",
    "Instruction",
}

BRITISH = re.compile(
    r"\b(colour|behaviour|normalis\w*|factoris\w*|parametris\w*|centre|"
    r"optimis\w*|discretis\w*|neighbour\w*|licence)\b",
    re.IGNORECASE,
)
# One reference-list entry: "[n] <authors, title> arxiv.org/abs/<id>". The
# `[^\[]` stops the match before the next "[m]" so inline citations elsewhere in
# the prose cannot bleed into the capture.
ARXIV = re.compile(r"\[(\d+)\]\s*([^\[]*?)https?://arxiv\.org/abs/([0-9a-z./-]+)")


def _text(node: ET.Element | None) -> str:
    return " ".join("".join(node.itertext()).split()) if node is not None else ""


def _raw(node: ET.Element | None) -> str:
    return ET.tostring(node, encoding="unicode") if node is not None else ""


def _base_names(cd: ET.Element) -> set[str]:
    return {b.text or "" for b in cd.findall("basecompoundref")}


def _check_compgraphs_class(cd: ET.Element, name: str) -> list[str]:
    """Light check for the in-scope compgraphs/ types: just a one-sentence brief
    and a detailed paragraph. No Inputs/Outputs/References, and no per-member
    sweep (these value types have many trivial overloaded constructors)."""
    gaps: list[str] = []
    brief = _text(cd.find("briefdescription"))
    detailed = _text(cd.find("detaileddescription"))
    if not brief:
        gaps.append("no brief")
    elif brief.count(".") == 0:
        gaps.append("brief is not a sentence")
    if not detailed:
        gaps.append("no detailed description")
    return gaps


def _check_instruction_yaml(path: str) -> list[str]:
    """Every instruction and every named input/output needs a non-empty desc."""
    try:
        import yaml
    except ImportError:
        return ["PyYAML not installed; cannot check instruction_set.yaml"]
    if not os.path.isfile(path):
        return [f"{path} not found"]
    gaps: list[str] = []
    with open(path) as fh:
        for section in yaml.safe_load_all(fh):
            for key, cmd in (section or {}).items():
                if key == "title":
                    continue
                if not str(cmd.get("desc") or "").strip():
                    gaps.append(f"{key}: no desc")
                for role in ("inputs", "outputs"):
                    entries = cmd.get(role)
                    if entries in ("any", None):
                        continue
                    for entry in entries:
                        if not str(entry.get("desc") or "").strip():
                            gaps.append(
                                f"{key}: {role[:-1]} `{entry.get('name')}` has no desc"
                            )
    return gaps


def main() -> int:
    files = sorted(glob.glob(os.path.join(XML_DIR, "*.xml")))
    if not files:
        sys.exit(f"no XML in {XML_DIR}; run `doxygen` from docs/ first")

    problems: dict[str, list[str]] = {}
    arxiv_strings: dict[str, set[str]] = {}

    for path in files:
        for cd in ET.parse(path).getroot().findall("compounddef"):
            if cd.get("kind") not in ("class", "struct"):
                continue
            if cd.get("prot") == "private":  # private nested implementation type
                continue
            loc = cd.find("location")
            f = loc.get("file", "") if loc is not None else ""
            name = cd.findtext("compoundname", "").removeprefix("madspace::")

            if "/compgraphs/" in f:
                if name in COMPGRAPHS_CLASSES:
                    g = _check_compgraphs_class(cd, name)
                    if g:
                        problems[name] = g
                continue
            if "/phasespace/" not in f:
                continue
            gaps: list[str] = []

            brief = _text(cd.find("briefdescription"))
            detailed_node = cd.find("detaileddescription")
            detailed = _text(detailed_node)
            raw = _raw(detailed_node)
            bases = _base_names(cd)
            is_component = bool(
                bases & {"madspace::Mapping", "madspace::FunctionGenerator"}
            )
            if not brief:
                gaps.append("no brief")
            elif brief.count(".") == 0:
                gaps.append("brief is not a sentence")
            # Mapping/FunctionGenerator subclasses carry the full comment; plain
            # helper structs only need a one-line brief.
            if is_component and not detailed:
                gaps.append("no detailed description")
            if not brief and not detailed:
                problems[name] = gaps
                continue
            if "madspace::Mapping" in bases:
                for kw in ("Inputs", "Conditions", "Outputs"):
                    if kw not in detailed:
                        gaps.append(f"missing **{kw}** list")
                if "returns a weight" not in detailed:
                    gaps.append("missing standing weight note")
            if "madspace::FunctionGenerator" in bases:
                for kw in ("Arguments", "Returns"):
                    if kw not in detailed:
                        gaps.append(f"missing **{kw}** list")
            if bases & {"madspace::Mapping", "madspace::FunctionGenerator"}:
                if "References" not in detailed:
                    gaps.append("missing **References** list")

            if "\\rst" in raw or "embed:rst" in raw:
                gaps.append("uses \\rst (breaks the Python docstring)")
            for m in re.finditer(r"`\s+-\s+`", detailed):
                gaps.append("hyphen instead of en-dash in a bullet")
                break
            # Scan prose only: quoted publication titles keep their own spelling.
            prose = detailed.split("References", 1)[0]
            brit = BRITISH.search(prose)
            if brit:
                gaps.append(f"British spelling: {brit.group(0)!r}")

            # Only the References list, so inline "[2]" markers are ignored.
            refs = raw.split("References", 1)[-1] if "References" in raw else ""
            for n, who, aid in ARXIV.findall(refs):
                who = re.sub(r"<[^>]+>", "", who)
                arxiv_strings.setdefault(aid, set()).add(" ".join(who.split())[:80])

            # public methods and constructor params
            for md in cd.iter("memberdef"):
                if md.get("prot") != "public":
                    continue
                kind = md.get("kind")
                mname = md.findtext("name", "")
                if kind == "function" and not mname.startswith("~"):
                    if not _text(md.find("briefdescription")) and not _text(
                        md.find("detaileddescription")
                    ):
                        gaps.append(f"undocumented public method: {mname}()")
                if kind == "function" and mname == name:  # a constructor
                    params = [p.findtext("declname", "") for p in md.findall("param")]
                    documented = {
                        pn.text
                        for pi in md.iter("parameteritem")
                        for pn in pi.iter("parametername")
                    }
                    for pn in params:
                        if pn and pn not in documented:
                            gaps.append(f"undocumented ctor param: {pn}")

            if gaps:
                problems[name] = gaps

    for aid, strings in sorted(arxiv_strings.items()):
        if len(strings) > 1:
            problems.setdefault("(references)", []).append(
                f"arXiv:{aid} cited with {len(strings)} different strings"
            )

    yaml_gaps = _check_instruction_yaml(YAML_PATH)
    if yaml_gaps:
        problems["instruction_set.yaml"] = yaml_gaps

    seen = {
        n
        for f in files
        for cd in ET.parse(f).getroot().findall("compounddef")
        if (n := cd.findtext("compoundname", "").removeprefix("madspace::"))
    }
    for missing in sorted(COMPGRAPHS_CLASSES - seen):
        problems.setdefault(missing, []).append(
            "expected compgraphs/ class not found in XML"
        )

    if not problems:
        print("check_doc_convention: all phasespace/ and compgraphs/ classes pass")
        return 0

    print(f"check_doc_convention: {len(problems)} class(es) with gaps\n")
    for name in sorted(problems):
        print(f"  {name}")
        for g in problems[name]:
            print(f"      - {g}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
