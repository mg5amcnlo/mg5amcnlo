# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "pyamplicol"
BUILTIN_ROOT = PACKAGE_ROOT / "models" / "builtin"

NON_RUNTIME_ALLOWLIST = (
    PACKAGE_ROOT / "assets",
    REPOSITORY_ROOT / "docs",
)

BUILTIN_IMPORT_ALLOWLIST = {
    "generation/lowering.py": {
        ("pyamplicol.models.builtin.lowering_reports", "module"),
        ("pyamplicol.models.builtin.lowering_tensor", "module"),
    },
    "generation/lowering_reports.py": {
        ("pyamplicol.models.builtin", "module"),
    },
    "generation/lowering_tensor.py": {
        ("pyamplicol.models.builtin", "module"),
    },
    "models/compiler.py": {
        ("pyamplicol.models.builtin.compiler", "function"),
    },
    "models/loading.py": {
        ("pyamplicol.models.builtin.adapters", "function"),
    },
    "generation/service.py": {
        ("pyamplicol.models.builtin.process_catalog", "function"),
        ("pyamplicol.models.builtin.process_ir", "function"),
        ("pyamplicol.models.builtin.process_selection", "function"),
        ("pyamplicol.models.builtin.process_types", "function"),
    },
}

LEGACY_PROCESS_SYMBOLS = frozenset(
    {
        "ALL_COLOURED",
        "ANTI_PARTICLE",
        "ANTIQUARKS",
        "CHARGES3",
        "FAMILY",
        "GLUONS",
        "PDGS",
        "QUARKS",
        "SINGLETS",
        "SORT_PARTICLES",
    }
)
LEGACY_PROCESS_SYMBOL_ALLOWLIST = {
    "generation/service.py": frozenset({"PDGS"}),
}
SM_PDG_VALUES = frozenset({*range(1, 7), *range(11, 17), 21, 22, 23, 24, 25, 26, 99})
BUILTIN_IMPLEMENTATION_DEFINITIONS = frozenset(
    {
        "BuiltinModel",
        "BuiltinSMDefinitionMixin",
        "BuiltinSMLoweringMixin",
        "BuiltinSMModel",
        "ProcessEnumerator",
        "_GraphTensorExpressionBuilder",
        "_build_auxiliary_tensor_probe",
        "_build_color_probe",
        "_charge3_sum",
        "_family_sum",
        "_legacy_selection_records_from_request",
        "_physical_pdgs",
        "build_interleaved_tensor_network_scalar_bundle",
        "build_symbolic_lowering_report",
        "build_tensor_network_scalar_bundle",
        "request_allows_charged_current",
    }
)


def _runtime_python_files() -> tuple[Path, ...]:
    return tuple(
        path
        for path in sorted(PACKAGE_ROOT.rglob("*.py"))
        if not path.is_relative_to(BUILTIN_ROOT)
        and not any(path.is_relative_to(root) for root in NON_RUNTIME_ALLOWLIST)
    )


def _relative(path: Path) -> str:
    return path.relative_to(PACKAGE_ROOT).as_posix()


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _resolved_import_from(path: Path, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    relative_parent = path.relative_to(PACKAGE_ROOT).parent
    package = ["pyamplicol", *relative_parent.parts]
    keep = len(package) - (node.level - 1)
    if keep < 1:
        return node.module or ""
    suffix = [] if node.module is None else node.module.split(".")
    return ".".join([*package[:keep], *suffix])


def _scope(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    parent = parents.get(node)
    while parent is not None:
        if isinstance(parent, (ast.AsyncFunctionDef, ast.FunctionDef)):
            return "function"
        parent = parents.get(parent)
    return "module"


def _pdg_expression(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        name = node.id.casefold()
        return (
            name == "pdg"
            or name.endswith("_pdg")
            or name.endswith("_pdg_code")
            or name in {"particle_id", "result_particle", "result_particle_id"}
        )
    if isinstance(node, ast.Attribute):
        return _pdg_expression(ast.Name(id=node.attr))
    if isinstance(node, ast.UnaryOp):
        return _pdg_expression(node.operand)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"abs", "int"}
        and len(node.args) == 1
        and not node.keywords
    ):
        return _pdg_expression(node.args[0])
    return False


def _comparison_integer_literals(node: ast.AST) -> frozenset[int]:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return frozenset({int(node.value)})
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _comparison_integer_literals(node.operand)
    if isinstance(node, (ast.List, ast.Set, ast.Tuple)):
        return frozenset().union(
            *(_comparison_integer_literals(item) for item in node.elts)
        )
    if isinstance(node, ast.Dict):
        return frozenset().union(
            *(_comparison_integer_literals(key) for key in node.keys if key is not None)
        )
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"abs", "int", "range"}
    ):
        return frozenset().union(
            *(_comparison_integer_literals(argument) for argument in node.args)
        )
    return frozenset()


@pytest.mark.parametrize(
    "module_name",
    (
        "pyamplicol.models.contracts",
        "pyamplicol.models.external",
        "pyamplicol.models.loading",
        "pyamplicol.processes.model",
    ),
)
def test_external_import_paths_do_not_load_builtin_model(module_name: str) -> None:
    script = "\n".join(
        (
            "import importlib",
            "import json",
            "import sys",
            f"sys.path.insert(0, {str(PACKAGE_ROOT.parent)!r})",
            f"importlib.import_module({module_name!r})",
            "loaded = sorted(name for name in sys.modules "
            "if name == 'pyamplicol.models.builtin' "
            "or name.startswith('pyamplicol.models.builtin.'))",
            "print(json.dumps(loaded))",
        )
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    loaded = json.loads(completed.stdout.strip().splitlines()[-1])
    assert loaded == [], (
        f"importing {module_name} crossed the built-in model boundary: {loaded}"
    )


def test_builtin_imports_are_limited_to_compatibility_and_dispatch_facades() -> None:
    violations: list[str] = []
    for path in _runtime_python_files():
        tree = _parse(path)
        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        relative = _relative(path)
        allowed = BUILTIN_IMPORT_ALLOWLIST.get(relative, set())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = _resolved_import_from(path, node)
                if not (
                    module == "pyamplicol.models.builtin"
                    or module.startswith("pyamplicol.models.builtin.")
                ):
                    continue
                import_scope = _scope(node, parents)
                if (module, import_scope) not in allowed:
                    violations.append(
                        f"{relative}:{node.lineno}: {module} ({import_scope})"
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if not (
                        alias.name == "pyamplicol.models.builtin"
                        or alias.name.startswith("pyamplicol.models.builtin.")
                    ):
                        continue
                    import_scope = _scope(node, parents)
                    if (alias.name, import_scope) not in allowed:
                        violations.append(
                            f"{relative}:{node.lineno}: {alias.name} ({import_scope})"
                        )

    assert violations == [], "unexpected built-in imports:\n" + "\n".join(violations)


def test_sm_catalog_and_pdg_classifiers_are_quarantined() -> None:
    catalog_violations: list[str] = []
    pdg_violations: list[str] = []
    for path in _runtime_python_files():
        tree = _parse(path)
        relative = _relative(path)
        allowed_symbols = LEGACY_PROCESS_SYMBOL_ALLOWLIST.get(relative, frozenset())
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in LEGACY_PROCESS_SYMBOLS:
                if node.id not in allowed_symbols:
                    catalog_violations.append(f"{relative}:{node.lineno}: {node.id}")
            elif isinstance(node, ast.ImportFrom):
                imported = LEGACY_PROCESS_SYMBOLS.intersection(
                    alias.name for alias in node.names
                )
                unexpected = imported.difference(allowed_symbols)
                if unexpected:
                    catalog_violations.append(
                        f"{relative}:{node.lineno}: {', '.join(sorted(unexpected))}"
                    )
            elif isinstance(node, ast.Compare):
                operands = (node.left, *node.comparators)
                if not any(_pdg_expression(operand) for operand in operands):
                    continue
                literals = frozenset().union(
                    *(
                        _comparison_integer_literals(operand)
                        for operand in operands
                        if not _pdg_expression(operand)
                    )
                )
                matched = literals.intersection(SM_PDG_VALUES)
                if matched:
                    pdg_violations.append(
                        f"{relative}:{node.lineno}: {sorted(matched)}"
                    )

    assert catalog_violations == [], "legacy process catalog escaped:\n" + "\n".join(
        catalog_violations
    )
    assert pdg_violations == [], "SM PDG classifier escaped:\n" + "\n".join(
        pdg_violations
    )


def test_builtin_implementation_uses_package_scoped_modules() -> None:
    required = {
        "definitions.py",
        "lowering.py",
        "lowering_reports.py",
        "lowering_tensor.py",
        "process_catalog.py",
        "process_enumeration.py",
        "process_ir.py",
        "process_selection.py",
        "process_types.py",
    }
    assert required.issubset({path.name for path in BUILTIN_ROOT.glob("*.py")})

    legacy_paths = (
        PACKAGE_ROOT / "models" / "builtin.py",
        PACKAGE_ROOT / "models" / "builtin_definitions.py",
        PACKAGE_ROOT / "models" / "builtin_lowering.py",
    )
    assert not any(path.exists() for path in legacy_paths)

    escaped_definitions: list[str] = []
    for path in _runtime_python_files():
        for node in ast.walk(_parse(path)):
            if (
                isinstance(
                    node,
                    (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef),
                )
                and node.name in BUILTIN_IMPLEMENTATION_DEFINITIONS
            ):
                escaped_definitions.append(
                    f"{_relative(path)}:{node.lineno}: {node.name}"
                )
    assert escaped_definitions == [], "built-in implementation escaped:\n" + "\n".join(
        escaped_definitions
    )
