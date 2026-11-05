#!/usr/bin/env python3
"""Generate a C++ docstring header from the Doxygen comments in the headers.

Doxygen is run over ``--include-dir`` to produce XML, which is turned into a
header exposing one ``consteval`` lookup, so the pybind11 bindings can attach
the C++ documentation without repeating it::

    namespace pd = madspace::pydoc;
    py::classh<FastRamboMapping, Mapping>(m, "FastRamboMapping", pd::doc("FastRamboMapping"))
        .def(py::init<...>(), pd::doc("FastRamboMapping::FastRamboMapping"), ...)
        .def("random_dim", &FastRamboMapping::random_dim, pd::doc("FastRamboMapping::random_dim"));

Keys are qualified names with the leading ``madspace::`` dropped. ``doc()`` is
``consteval``: an unknown key is a compile-time error at the call site, and
there is no run-time cost. Undocumented but existing entities are registered
with an empty string, so referencing them still builds.

If Doxygen is missing or fails, a stub header is written whose ``doc()`` always
returns ``""``; the build then succeeds without docstrings.
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

STRIP_NAMESPACE = "madspace::"

DOXYFILE_TEMPLATE = """\
PROJECT_NAME    = madspace
JAVADOC_AUTOBRIEF = YES
INPUT          = {include_dir}
RECURSIVE      = YES
EXCLUDE_PATTERNS = *mixin*
GENERATE_HTML   = NO
GENERATE_LATEX  = NO
GENERATE_XML    = YES
XML_OUTPUT      = {xml_out}
QUIET          = YES
WARNINGS       = NO
WARN_IF_UNDOCUMENTED = NO
"""


def strip_ns(name: str) -> str:
    """`madspace::FastRamboMapping::random_dim` -> `FastRamboMapping::random_dim`."""
    return name[len(STRIP_NAMESPACE) :] if name.startswith(STRIP_NAMESPACE) else name


# Doxygen inline markup -> reStructuredText, so Sphinx renders the Python
# docstrings with the same emphasis/monospace as the C++ (Breathe) side.
_INLINE_WRAP = {
    "computeroutput": ("``", "``"),
    "verbatim": ("``", "``"),
    "emphasis": ("*", "*"),
    "bold": ("**", "**"),
}


def _formula_rst(tex: str) -> str:
    """Doxygen <formula> body (\\f$..\\f$ / \\f[..\\f]) -> reStructuredText math."""
    tex = tex.strip()
    if tex.startswith("$") and tex.endswith("$") and not tex.startswith("$$"):
        return f":math:`{tex.strip('$').strip()}`"
    for pre, suf in (("\\[", "\\]"), ("$$", "$$")):
        if tex.startswith(pre) and tex.endswith(suf):
            tex = tex[len(pre) : -len(suf)]
    body = "\n   ".join(line.strip() for line in tex.strip().splitlines())
    return f"\n\n.. math::\n\n   {body}\n\n"


def text_of(node: ET.Element) -> str:
    """Flatten a Doxygen description subtree into reStructuredText."""
    parts = []

    def walk(el, prefix=""):
        tag = el.tag
        if tag == "para":
            # Blank line between paragraphs, but not right after a bullet opened.
            if not (parts and parts[-1].endswith("- ")):
                parts.append("\n\n")
        elif tag == "parameterlist":
            parts.append(
                "\n" + ("Args:" if el.get("kind") == "param" else "Returns:") + "\n"
            )
        elif tag == "parameteritem":
            names = [n.text or "" for n in el.iter("parametername")]
            desc = "".join(
                text_of(pd) for pd in el.iter("parameterdescription")
            ).strip()
            parts.append(f"    {', '.join(names)}: {desc}\n")
            return  # handled the whole item
        elif tag == "simplesect":
            kind = el.get("kind", "")
            parts.append("\n" + kind.capitalize() + ": ")
        elif tag == "listitem":
            parts.append("\n" + prefix + "- ")
        elif tag == "formula":
            parts.append(_formula_rst(el.text or ""))
            return  # no children, tail handled by caller

        open_m, close_m = _INLINE_WRAP.get(tag, ("", ""))
        if open_m:
            parts.append(open_m)
        if el.text:
            parts.append(el.text)
        for child in el:
            walk(child)
            if child.tail:
                parts.append(child.tail)
        if close_m:
            parts.append(close_m)

    walk(node)
    raw = "".join(parts)
    # Collapse the whitespace Doxygen's pretty-printing leaves behind, but keep
    # the leading indent so parameter lists stay readable.
    lines = []
    for ln in raw.splitlines():
        indent, rest = re.match(r"(\s*)(.*)", ln).groups()
        lines.append(indent + re.sub(r"[ \t]+", " ", rest).rstrip())
    out = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def description(memberdef: ET.Element) -> str:
    brief = memberdef.find("briefdescription")
    detailed = memberdef.find("detaileddescription")
    chunks = [text_of(brief) if brief is not None else ""]
    chunks.append(text_of(detailed) if detailed is not None else "")
    return "\n\n".join(c for c in chunks if c).strip()


def collect(xml_dir: Path) -> dict[str, str]:
    """Lookup key -> docstring, for every class/struct/namespace member."""
    docs: dict[str, str] = {}

    def register(key: str, text: str) -> None:
        if key not in docs:
            docs[key] = text
        elif docs[key] != text and text:
            # Overloads with genuinely different docs: disambiguate by count.
            n = 2
            while f"{key}#{n}" in docs:
                n += 1
            docs[f"{key}#{n}"] = text

    for xml_file in sorted(xml_dir.glob("*.xml")):
        if xml_file.name in ("index.xml", "Doxyfile.xml"):
            continue
        root = ET.parse(xml_file).getroot()
        for compound in root.findall("compounddef"):
            kind = compound.get("kind")
            if kind not in ("class", "struct", "namespace"):
                continue
            cname = strip_ns(compound.findtext("compoundname", ""))
            if kind in ("class", "struct"):
                register(cname, description(compound))
            for member in compound.iter("memberdef"):
                mname = member.findtext("name", "")
                if not mname or mname.startswith("~"):
                    continue
                prefix = f"{cname}::" if kind in ("class", "struct") else ""
                register(f"{prefix}{mname}", description(member))
    return docs


def render(docs: dict[str, str]) -> str:
    entries = []
    for key in sorted(docs):
        # PYDOC delimiter keeps a stray )PYDOC" in a comment from ending the literal.
        body = docs[key].replace(')PYDOC"', ') PYDOC"')
        entries.append(f'    {{{cpp_str(key)}, R"PYDOC({body})PYDOC"}},')
    table = "\n".join(entries)
    return f"""\
// Generated by madspace/generate_docstrings.py from Doxygen comments.
// Do not edit by hand.
#pragma once

#include <string_view>
#include <utility>

namespace madspace::pydoc {{

namespace detail {{
inline constexpr std::pair<std::string_view, std::string_view> table[] = {{
{table}
}};
}} // namespace detail

/// C++ doc comment for a qualified name with the leading "madspace::" dropped
/// (e.g. "FastRamboMapping::random_dim"). consteval: an unknown key is a
/// compile error at the call site and there is no run-time cost.
consteval const char* doc(std::string_view key) {{
    for (const auto& [k, v] : detail::table) {{
        if (k == key) {{
            return v.data();
        }}
    }}
    throw "generate_docstrings: no doc entry for this key";
}}

}} // namespace madspace::pydoc
"""


def cpp_str(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def stub_header(reason: str) -> str:
    return f"""\
// Generated by madspace/generate_docstrings.py -- empty docstrings.
// Reason: {reason}
#pragma once

#include <string_view>

namespace madspace::pydoc {{
consteval const char* doc(std::string_view) {{ return ""; }}
}} // namespace madspace::pydoc
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-dir", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--doxygen", default="doxygen")
    parser.add_argument(
        "--require-doxygen",
        action="store_true",
        help="fail instead of writing an empty stub when doxygen is unavailable "
        "(use in release/wheel builds so docstrings are never silently dropped)",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="skip doxygen entirely and write the empty-docs stub header "
        "(used when documentation generation is disabled)",
    )
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.stub:
        args.out.write_text(stub_header("documentation generation disabled"))
        print(f"generate_docstrings: wrote stub header to {args.out}")
        return 0
    if args.include_dir is None:
        parser.error("--include-dir is required unless --stub is given")

    def give_up(reason: str) -> int:
        if args.require_doxygen:
            print(f"generate_docstrings: {reason}", file=sys.stderr)
            return 1
        args.out.write_text(stub_header(reason))
        print(f"generate_docstrings: {reason}, wrote stub header", file=sys.stderr)
        return 0

    doxygen = shutil.which(args.doxygen)
    if doxygen is None:
        return give_up("doxygen not found on PATH")

    with tempfile.TemporaryDirectory() as tmp:
        xml_out = Path(tmp) / "xml"
        doxyfile = DOXYFILE_TEMPLATE.format(
            include_dir=args.include_dir.resolve(), xml_out=xml_out
        )
        proc = subprocess.run(
            [doxygen, "-"],
            input=doxyfile,
            text=True,
            cwd=tmp,
            capture_output=True,
        )
        if proc.returncode != 0 or not xml_out.is_dir():
            sys.stderr.write(proc.stderr)
            return give_up("doxygen run failed")
        docs = collect(xml_out)

    args.out.write_text(render(docs))
    print(f"generate_docstrings: wrote {len(docs)} docstrings to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
