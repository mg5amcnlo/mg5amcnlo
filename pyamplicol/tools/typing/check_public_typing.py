# SPDX-License-Identifier: 0BSD
"""Validate and type-check the installed public Python contract."""

from __future__ import annotations

import ast
import importlib
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
SOURCE_PACKAGE = ROOT / "src" / "pyamplicol"
RUST_STUB = (
    ROOT
    / "rust"
    / "crates"
    / "rusticol-python"
    / "stubs"
    / "pyamplicol"
    / "_rusticol.pyi"
)
RUST_BINDING = ROOT / "rust" / "crates" / "rusticol-python" / "src" / "lib.rs"
BUILD_BACKEND = ROOT / "build_backend" / "_pyamplicol_build.py"
PUBLIC_CONSUMER = ROOT / "tests" / "typing" / "consumer_public_api.py"
NATIVE_CONSUMER = ROOT / "tests" / "typing" / "consumer_native_api.py"

PUBLIC_FACADES: dict[str, Path] = {
    "pyamplicol": SOURCE_PACKAGE / "__init__.py",
    "pyamplicol.api": SOURCE_PACKAGE / "api" / "__init__.py",
    "pyamplicol.config": SOURCE_PACKAGE / "config" / "__init__.py",
    "pyamplicol.reporting": SOURCE_PACKAGE / "reporting" / "__init__.py",
    "pyamplicol.runtime": SOURCE_PACKAGE / "runtime" / "__init__.py",
}


@dataclass(frozen=True, slots=True)
class MetadataCheck:
    errors: tuple[str, ...]
    native_symbols: tuple[str, ...] | None


def static_all(path: Path) -> tuple[str, ...]:
    """Read one literal ``__all__`` declaration without importing the package."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, (ast.List, ast.Tuple)):
            raise ValueError(f"{path}: __all__ must be a literal list or tuple")
        exports = tuple(
            item.value
            for item in node.value.elts
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        )
        if len(exports) != len(node.value.elts):
            raise ValueError(f"{path}: __all__ must contain only string literals")
        if len(exports) != len(set(exports)):
            raise ValueError(f"{path}: __all__ contains duplicate names")
        return exports
    raise ValueError(f"{path}: no static __all__ declaration")


def _stub_definitions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    }


def _binding_exports(path: Path) -> set[str]:
    source = path.read_text(encoding="utf-8")
    classes = set(re.findall(r"module\.add_class::<([A-Za-z_][A-Za-z0-9_]*)>", source))
    functions = set(
        re.findall(r"wrap_pyfunction!\(([A-Za-z_][A-Za-z0-9_]*),\s*module\)", source)
    )
    functions = {name for name in functions if not name.startswith("_")}
    exceptions = set(re.findall(r'module\.add\(\s*"([A-Za-z_][A-Za-z0-9_]*)"', source))
    return classes | functions | exceptions


def _backend_stages_stub(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    stage_function: ast.FunctionDef | None = None
    call_count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_stage_python_stub":
            stage_function = node
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_stage_python_stub"
        ):
            call_count += 1
    if stage_function is None or call_count == 0:
        return False
    constants = {
        node.value
        for node in ast.walk(stage_function)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    copies_file = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "copy2"
        for node in ast.walk(stage_function)
    )
    return "_rusticol.pyi" in constants and copies_file


def check_typing_metadata() -> MetadataCheck:
    """Return metadata/stub contract failures without rebuilding the extension."""

    errors: list[str] = []
    marker = SOURCE_PACKAGE / "py.typed"
    if not marker.is_file():
        errors.append(f"missing inline typing marker: {marker.relative_to(ROOT)}")
    elif marker.read_text(encoding="utf-8").strip():
        errors.append("py.typed must be an empty full-package marker")

    if not RUST_STUB.is_file():
        errors.append(f"missing maintained native stub: {RUST_STUB.relative_to(ROOT)}")
        return MetadataCheck(tuple(errors), None)
    if not _backend_stages_stub(BUILD_BACKEND):
        errors.append("build backend no longer stages the maintained _rusticol.pyi")

    stub_symbols = _stub_definitions(RUST_STUB)
    binding_symbols = _binding_exports(RUST_BINDING)
    missing_stub_symbols = sorted(binding_symbols - stub_symbols)
    extra_stub_symbols = sorted(stub_symbols - binding_symbols)
    if missing_stub_symbols:
        errors.append(
            "native stub misses Rust exports: " + ", ".join(missing_stub_symbols)
        )
    if extra_stub_symbols:
        errors.append(
            "native stub declares non-exported symbols: "
            + ", ".join(extra_stub_symbols)
        )

    native_symbols: tuple[str, ...] | None = None
    try:
        native = importlib.import_module("pyamplicol._rusticol")
    except (ImportError, OSError):
        pass
    else:
        native_symbols = tuple(
            sorted(name for name in dir(native) if not name.startswith("_"))
        )
        missing_runtime_symbols = sorted(set(native_symbols) - stub_symbols)
        if missing_runtime_symbols:
            errors.append(
                "native stub misses exports in the available extension: "
                + ", ".join(missing_runtime_symbols)
            )
    return MetadataCheck(tuple(errors), native_symbols)


def render_export_consumer() -> str:
    """Build a consumer expression for every current public facade export."""

    aliases = {
        "pyamplicol": "package",
        "pyamplicol.api": "api",
        "pyamplicol.config": "config",
        "pyamplicol.reporting": "reporting",
        "pyamplicol.runtime": "runtime",
    }
    lines = [
        "# Generated in a temporary directory by the public typing gate.",
        "from __future__ import annotations",
        "",
        "from typing import reveal_type",
        "",
    ]
    for module, alias in aliases.items():
        lines.append(f"import {module} as {alias}")
    lines.append("")
    for module, path in PUBLIC_FACADES.items():
        alias = aliases[module]
        lines.extend(f"reveal_type({alias}.{name})" for name in static_all(path))
    lines.append("")
    return "\n".join(lines)


def _stage_installed_package(destination: Path) -> Path:
    site_packages = destination / "site-packages"
    installed = site_packages / "pyamplicol"
    shutil.copytree(
        SOURCE_PACKAGE,
        installed,
        ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", "*.so", "ufo", "api_templates"
        ),
    )
    shutil.copy2(RUST_STUB, installed / "_rusticol.pyi")
    return site_packages


def _run_mypy(command: list[str], *, environment: dict[str, str] | None = None) -> int:
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    return completed.returncode


def _check_installed_consumers() -> int:
    with TemporaryDirectory(prefix="pyamplicol-typing-") as temporary:
        root = Path(temporary)
        site_packages = _stage_installed_package(root)
        generated_consumer = root / "consumer_all_exports.py"
        generated_consumer.write_text(render_export_consumer(), encoding="utf-8")
        config = root / "mypy.ini"
        config.write_text(
            "\n".join(
                (
                    "[mypy]",
                    "python_version = 3.11",
                    "strict = True",
                    "follow_imports = silent",
                    f"mypy_path = {site_packages}",
                    "no_site_packages = True",
                    "show_error_codes = True",
                    "warn_unused_configs = True",
                    "",
                )
            ),
            encoding="utf-8",
        )
        environment = os.environ.copy()
        environment.pop("MYPYPATH", None)
        consumer_status = _run_mypy(
            [
                sys.executable,
                "-m",
                "mypy",
                "--config-file",
                str(config),
                str(PUBLIC_CONSUMER),
                str(NATIVE_CONSUMER),
            ],
            environment=environment,
        )
        if consumer_status != 0:
            return consumer_status

        exports = subprocess.run(
            [
                sys.executable,
                "-m",
                "mypy",
                "--config-file",
                str(config),
                str(generated_consumer),
            ],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        if exports.returncode != 0:
            sys.stdout.write(exports.stdout)
            sys.stderr.write(exports.stderr)
            return exports.returncode
        reveal_lines = tuple(
            line for line in exports.stdout.splitlines() if "Revealed type is" in line
        )
        expected_reveals = sum(
            len(static_all(path)) for path in PUBLIC_FACADES.values()
        )
        if len(reveal_lines) != expected_reveals:
            print(
                "typing export error: expected "
                f"{expected_reveals} revealed types, found {len(reveal_lines)}",
                file=sys.stderr,
            )
            return 1
        degraded = tuple(
            line
            for line in reveal_lines
            if re.search(r'Revealed type is ["\']Any["\']$', line)
        )
        if degraded:
            print(
                "typing export error: public exports degraded to Any",
                file=sys.stderr,
            )
            print("\n".join(degraded), file=sys.stderr)
            return 1
        return 0


def main() -> int:
    metadata = check_typing_metadata()
    for error in metadata.errors:
        print(f"typing metadata error: {error}", file=sys.stderr)
    if metadata.errors:
        return 1
    if metadata.native_symbols is None:
        print(
            "typing metadata: native extension unavailable; "
            "checked canonical Rust exports"
        )
    else:
        print(
            "typing metadata: covered "
            f"{len(metadata.native_symbols)} available native exports"
        )

    source_status = _run_mypy([sys.executable, "-m", "mypy"])
    if source_status != 0:
        return source_status
    return _check_installed_consumers()


if __name__ == "__main__":
    raise SystemExit(main())
