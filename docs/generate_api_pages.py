"""Generate one Sphinx page per madspace class or free function, for the C++
and Python APIs.

Run from conf.py after the Doxygen XML is produced. Writes
``<source>/madspace/cpp/<Class>.rst`` (breathe ``doxygenclass`` /
``doxygenstruct`` / ``doxygenfunction``) and
``<source>/madspace/python/<Class>.rst`` (``autoclass`` / ``autofunction``),
then rewrites the ``cpp-api.rst`` / ``python-api.rst`` landing pages as a
grouped overview (mappings, function generators, compute graph, ...). Both
output directories are wiped first so a removed class or function does not
leave a stale page behind.

Free functions are limited to the ones bound into the Python module: that is
already the curated list of top-level ``madspace`` functions meant for
end users, as opposed to internal helpers (``operator<<``, ``to_json``, ...)
that only exist for the C++ implementation.
"""

import argparse
import inspect
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

# Top-level classes in the madspace namespace only; nested types (Foo::Bar)
# are documented within their parent's page via :members:.
_TOPLEVEL = re.compile(r"^madspace::[A-Za-z_][A-Za-z0-9_]*$")

_CPP_OPTIONS = (":members:", ":undoc-members:")
_PY_OPTIONS = (":members:", ":undoc-members:", ":show-inheritance:")

# Implementation detail types that should not get their own page. Every
# ``*Instruction`` subclass is excluded too (see ``_denied``); the base
# ``Instruction`` is kept.
_DENY_EXACT = {
    "ShapeExpr",
    "Opcode",
    "InstructionDependencies",
    "LastUseOfLocals",
}


def _denied(short: str) -> bool:
    if short in _DENY_EXACT:
        return True
    return short.endswith("Instruction") and short != "Instruction"


@dataclass
class _Compound:
    short: str
    full: str
    kind: str
    loc: str = ""
    bases: set = field(default_factory=set)


# (heading, one-line intro, predicate). First match wins; the predicate sees a
# ``_Compound`` for the C++ side and a class object for the Python side.
_CPP_GROUPS = [
    (
        "Phase-space mappings",
        "Invertible maps between the unit hypercube and physical phase space; "
        "each also returns its Jacobian weight.",
        lambda c: "madspace::Mapping" in c.bases or c.short == "Mapping",
    ),
    (
        "Function generators",
        "Builders that assemble compute-graph functions for channel weights, "
        "energy scales and network outputs.",
        lambda c: "madspace::FunctionGenerator" in c.bases
        or c.short == "FunctionGenerator",
    ),
    (
        "Phase-space building blocks",
        "Supporting types shared by the mappings: topologies, cuts, observables "
        "and density grids.",
        lambda c: "/phasespace/" in c.loc,
    ),
    (
        "Compute graph",
        "The typed value model and the builder API used to record and compile "
        "functions from instructions.",
        lambda c: "/compgraphs/" in c.loc,
    ),
    (
        "Driver and runtime",
        "The high-level driver plus the tensor and device layer it runs on: "
        "integration, unweighting, event output and execution.",
        lambda c: "/driver/" in c.loc,
    ),
]
_OTHER_GROUP = ("Utilities", "Lower-level helpers and containers.")

# Python-only classes with no top-level C++ counterpart (nested C++ types
# exposed to Python as their own class, plus a couple of pure-Python
# wrappers). Mapped to the C++ class whose category they should inherit.
_PY_ALIAS_PARENT = {
    "CachedPdf": "DifferentialCrossSection",
    "CachedScale": "DifferentialCrossSection",
    "CutItem": "Cuts",
    "Decay": "Topology",
    "HistItem": "ObservableHistograms",
    "LineRef": "Diagram",
    "MadnisConfig": "MadnisTraining",
    "NamedTypes": "NamedVector",
    "NamedValues": "NamedVector",
    "SubprocArgs": "LHECompleter",
    "TrainingArgs": "MultiMadnisTraining",
    "Verbosity": "Logger",
}


def _py_groups(mod, cpp_category):
    mapping = getattr(mod, "Mapping", ())
    generator = getattr(mod, "FunctionGenerator", ())
    compgraph = {
        "Type",
        "Value",
        "BatchSize",
        "DataType",
        "Function",
        "FunctionBuilder",
        "FunctionRuntime",
        "InstructionCall",
        "Instruction",
    }

    def in_cpp_group(heading):
        return lambda obj: (
            cpp_category.get(_PY_ALIAS_PARENT.get(obj.__name__, obj.__name__))
            == heading
        )

    return [
        (
            _CPP_GROUPS[0][0],
            _CPP_GROUPS[0][1],
            lambda obj: (
                isinstance(obj, type)
                and isinstance(mapping, type)
                and issubclass(obj, mapping)
            )
            or in_cpp_group(_CPP_GROUPS[0][0])(obj),
        ),
        (
            _CPP_GROUPS[1][0],
            _CPP_GROUPS[1][1],
            lambda obj: (
                isinstance(obj, type)
                and isinstance(generator, type)
                and issubclass(obj, generator)
            )
            or in_cpp_group(_CPP_GROUPS[1][0])(obj),
        ),
        (
            _CPP_GROUPS[2][0],
            _CPP_GROUPS[2][1],
            in_cpp_group(_CPP_GROUPS[2][0]),
        ),
        (
            _CPP_GROUPS[3][0],
            _CPP_GROUPS[3][1],
            lambda obj: obj.__name__ in compgraph
            or in_cpp_group(_CPP_GROUPS[3][0])(obj),
        ),
        (
            _CPP_GROUPS[4][0],
            _CPP_GROUPS[4][1],
            in_cpp_group(_CPP_GROUPS[4][0]),
        ),
    ]


def _page(title: str, directive: str, target: str, options) -> str:
    body = "".join(f"   {opt}\n" for opt in options)
    return f"{title}\n{'=' * len(title)}\n\n.. {directive}:: {target}\n{body}"


_EDIT_WARNING = ".. Generated by docs/generate_api_pages.py - do not edit by hand."


def _landing(title: str, intro: str, subdir: str, groups) -> str:
    """``groups`` is a list of (heading, intro, [page names])."""
    out = [_EDIT_WARNING, "", title, "=" * len(title), "", intro, ""]
    for heading, gintro, names in groups:
        if not names:
            continue
        out += [
            heading,
            "-" * len(heading),
            "",
            gintro,
            "",
            ".. toctree::",
            "   :maxdepth: 1",
            "",
        ]
        out += [f"   {subdir}/{name}" for name in sorted(names)]
        out += [""]
    return "\n".join(out)


def _landing_fallback(title: str, intro: str, subdir: str) -> str:
    """Used when there is no Doxygen XML / no importable module (e.g. a plain
    Read the Docs build): a globbed toctree so the page never dangles."""
    return "\n".join(
        [
            _EDIT_WARNING,
            "",
            title,
            "=" * len(title),
            "",
            intro,
            "",
            ".. toctree::",
            "   :glob:",
            "   :maxdepth: 1",
            "",
            f"   {subdir}/*",
            "",
        ]
    )


def _replace_dir(path: Path, pages: dict) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    for name, text in pages.items():
        (path / f"{name}.rst").write_text(text)


def _cpp_compounds(xml_dir: Path):
    index = xml_dir / "index.xml"
    if not index.is_file():
        print(f"generate_api_pages: {index} missing; skipping C++ API")
        return []
    root = ET.parse(index).getroot()
    found = []
    for compound in root.findall("compound"):
        kind = compound.get("kind")
        name = compound.findtext("name", "")
        if kind not in ("class", "struct") or not _TOPLEVEL.match(name):
            continue
        short = name.split("::")[-1]
        if _denied(short):
            continue
        loc, bases = "", set()
        cfile = xml_dir / f"{compound.get('refid')}.xml"
        if cfile.is_file():
            cd = ET.parse(cfile).getroot().find("compounddef")
            if cd is not None:
                node = cd.find("location")
                loc = node.get("file", "") if node is not None else ""
                bases = {b.text or "" for b in cd.findall("basecompoundref")}
        found.append(_Compound(short, name, kind, loc, bases))
    return sorted(found, key=lambda c: c.short)


def _cpp_namespace_functions(xml_dir: Path):
    """Free functions declared directly in the top-level ``madspace``
    namespace, keyed by short name. Used to look up the C++ declaration (for
    its doxygenfunction page and source location) of a function bound into
    the Python module."""
    ns_file = xml_dir / "namespacemadspace.xml"
    if not ns_file.is_file():
        return {}
    cd = ET.parse(ns_file).getroot().find("compounddef")
    found = {}
    for sec in cd.findall("sectiondef"):
        for md in sec.findall("memberdef"):
            if md.get("kind") != "function":
                continue
            name = md.findtext("name", "")
            if not name or name in found:
                continue
            node = md.find("location")
            loc = node.get("file", "") if node is not None else ""
            found[name] = _Compound(name, f"madspace::{name}", "function", loc)
    return found


def _python_classes():
    try:
        import madspace
    except Exception as exc:  # noqa: BLE001 - want the doc build to continue
        print(
            f"generate_api_pages: cannot import madspace ({exc}); skipping Python API"
        )
        return None, []
    members = [
        (name, obj)
        for name, obj in inspect.getmembers(madspace, inspect.isclass)
        if getattr(obj, "__module__", "").startswith("madspace")
        and not name.startswith("_")
        and not _denied(name)
    ]
    return madspace, sorted(members)


def _python_functions(mod):
    if mod is None:
        return []
    return sorted(
        (name, obj)
        for name, obj in inspect.getmembers(mod, inspect.isroutine)
        if getattr(obj, "__module__", "").startswith("madspace")
        and not name.startswith("_")
    )


def _assign(items, groups, key):
    """Bucket ``items`` into ``groups`` by first matching predicate; the rest
    go to ``_OTHER_GROUP``. Returns [(heading, intro, [names])]."""
    buckets = {heading: [] for heading, _, _ in groups}
    other = []
    for item in items:
        for heading, _, pred in groups:
            if pred(item):
                buckets[heading].append(key(item))
                break
        else:
            other.append(key(item))
    result = [(h, i, buckets[h]) for h, i, _ in groups]
    result.append((_OTHER_GROUP[0], _OTHER_GROUP[1], other))
    return result


def generate_api_pages(source_dir, xml_dir) -> None:
    base = Path(source_dir) / "madspace"

    compounds = _cpp_compounds(Path(xml_dir))
    mod, members = _python_classes()
    functions = _python_functions(mod)

    ns_functions = _cpp_namespace_functions(Path(xml_dir))
    function_compounds = [
        ns_functions[name] for name, _ in functions if name in ns_functions
    ]
    cpp_items = compounds + function_compounds

    cpp_pages = {
        c.short: _page(
            c.short,
            {"struct": "doxygenstruct", "function": "doxygenfunction"}.get(
                c.kind, "doxygenclass"
            ),
            c.full,
            () if c.kind == "function" else _CPP_OPTIONS,
        )
        for c in cpp_items
    }
    _replace_dir(base / "cpp", cpp_pages)
    cpp_title = "C++ API"
    cpp_intro = (
        "Reference for the ``madspace`` C++ classes and free functions, "
        "extracted from the header comments, one page per class or function."
    )
    cpp_grouped = _assign(cpp_items, _CPP_GROUPS, lambda c: c.short)
    cpp_category = {
        name: heading for heading, _, names in cpp_grouped for name in names
    }
    (base / "cpp-api.rst").write_text(
        _landing(cpp_title, cpp_intro, "cpp", cpp_grouped)
        if cpp_items
        else _landing_fallback(cpp_title, cpp_intro, "cpp")
    )

    py_pages = {
        name: _page(name, "autoclass", f"madspace.{name}", _PY_OPTIONS)
        for name, _ in members
    }
    py_pages.update(
        {
            name: _page(name, "autofunction", f"madspace.{name}", ())
            for name, _ in functions
        }
    )
    _replace_dir(base / "python", py_pages)
    py_title = "Python API"
    py_intro = (
        "Reference for the classes and free functions exposed by the "
        ":mod:`madspace` Python module, one page per class or function."
    )
    py_items = [obj for _, obj in members] + [obj for _, obj in functions]
    (base / "python-api.rst").write_text(
        _landing(
            py_title,
            py_intro,
            "python",
            _assign(
                py_items,
                _py_groups(mod, cpp_category),
                lambda obj: obj.__name__,
            ),
        )
        if py_items
        else _landing_fallback(py_title, py_intro, "python")
    )

    print(
        f"generate_api_pages: wrote {len(cpp_pages)} C++ and "
        f"{len(py_pages)} Python pages"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="source", type=Path)
    parser.add_argument("--xml-dir", default="build/doxygenxml", type=Path)
    args = parser.parse_args()
    generate_api_pages(args.source_dir, args.xml_dir)
