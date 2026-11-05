# SPDX-License-Identifier: 0BSD
"""Opt-in, tiny compiler regressions; no pyAmpliCol native build is required.

PYAMPLICOL_RUN_SYMJIT_REGRESSIONS=1 tests the locked published SymJIT version.
PYAMPLICOL_SYMJIT_SOURCE=/path/to/symjit selects a different source checkout.
Only test modules are added to a temporary copy: corrective patches are never
applied. A broken compiler is expected to fail, rather than produce an xfail.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import signal
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FOLLOWUP = ROOT / "SYMJIT_FOLLOW_UP_FIXES"
ORIGINAL = ROOT / "DEPENDENCY_FIXES" / "SYMJIT"
_ARM = platform.machine().lower() in {"arm64", "aarch64"}
pytestmark = pytest.mark.skipif(
    os.environ.get("PYAMPLICOL_RUN_SYMJIT_REGRESSIONS") != "1",
    reason="set PYAMPLICOL_RUN_SYMJIT_REGRESSIONS=1 for SymJIT compiler tests",
)

_BINARIES = {
    "output_reuse": ORIGINAL / "output-reuse/src/main.rs",
    "unused_trailing_inputs": ORIGINAL / "unused-trailing-inputs/src/main.rs",
    "argument_count": FOLLOWUP / "argument-count-overflow/mre.rs",
    "arm_simd_arguments": FOLLOWUP / "arm-simd-real-ultra-arguments/mre.rs",
    "real_input_locations": FOLLOWUP / "real-input-locations/mre.rs",
}
_RUST_TESTS = {
    "compressed_success": ORIGINAL / "compressed-success-status/mre.rs",
    "original_arm_registers": ORIGINAL / "arm-callee-saved-registers/mre.rs",
    "arm_normal_return": FOLLOWUP / "arm-compressed-register-restore/mre.rs",
    "arm_fast_complex": FOLLOWUP
    / "arm-compressed-register-restore/fast_complex_mre.rs",
}


def _cargo(manifest: Path, *arguments: str) -> str:
    """Keep compiler crashes isolated and terminate a timed-out process group."""
    command = ["cargo", *arguments, "--manifest-path", str(manifest)]
    # Cargo arguments must precede the test/program's own `--` arguments.
    if "--" in arguments:
        split = command.index("--")
        command[split:split] = command[-2:]
        del command[-2:]
    environment = os.environ.copy()
    environment.setdefault(
        "CARGO_TARGET_DIR", str(ROOT / ".artifacts/symjit-regressions/target")
    )
    environment["CARGO_BUILD_JOBS"] = "2"
    with subprocess.Popen(
        command,
        cwd=manifest.parent,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=os.name == "posix",
    ) as process:
        try:
            output, errors = process.communicate(timeout=600)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            output, errors = process.communicate()
            pytest.fail(f"Cargo timed out after 600 seconds:\n{output}{errors}")
        assert process.returncode == 0, (
            f"{' '.join(command)} exited {process.returncode}:\n{output}{errors}"
        )
    return output


def _manifest(dependency: str) -> str:
    text = (
        '[package]\nname="pyamplicol-symjit-regressions"\n'
        'version="0.0.0"\nedition="2021"\npublish=false\n'
        '[workspace]\n[features]\ndefault=["symbolica"]\n'
        'symbolica=["symjit/symbolica"]\n[dependencies]\nanyhow="1"\n'
        f"symjit={{ {dependency}, default-features=false }}\n"
    )
    for kind, targets in (("bin", _BINARIES), ("test", _RUST_TESTS)):
        for name, source in targets.items():
            text += (
                f"[[{kind}]]\nname={json.dumps(name)}\npath={json.dumps(str(source))}\n"
            )
    return text


@pytest.fixture(scope="session")
def symjit_manifests(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    assert shutil.which("cargo"), "these opt-in tests require Rust/Cargo"
    work = tmp_path_factory.mktemp("symjit-regressions")
    # Keep the harness beside, not above, a checkout with its own workspace.
    harness = work / "harness" / "Cargo.toml"
    harness.parent.mkdir()
    selected = os.environ.get("PYAMPLICOL_SYMJIT_SOURCE")
    with (ROOT / "dependencies/release-lock.toml").open("rb") as stream:
        version = tomllib.load(stream)["symjit"]["version"]
    dependency = (
        f"path={json.dumps(str(Path(selected).expanduser().resolve()))}"
        if selected
        else f'version="={version}"'
    )
    harness.write_text(_manifest(dependency))
    metadata = json.loads(_cargo(harness, "metadata", "--format-version", "1"))
    package = next(p for p in metadata["packages"] if p["name"] == "symjit")
    source = Path(package["manifest_path"]).parent
    print(f"\nSymJIT {package['version']}: {source}")

    staged = work / "symjit"
    shutil.copytree(source, staged, ignore=shutil.ignore_patterns(".git", "target"))
    # Both tests need private lowering types. Append test modules, not fixes,
    # to the staged source; the selected checkout/registry copy is untouched.
    for relative, module, test_source in (
        (
            "src/symjit/mod.rs",
            "pyamplicol_return_regression",
            FOLLOWUP / "compressed-complex-return/mre.rs",
        ),
        (
            "src/symjit/complexify.rs",
            "pyamplicol_abs_regression",
            FOLLOWUP / "real-input-locations/abs_mre.rs",
        ),
    ):
        target = staged / relative
        target.write_text(
            target.read_text()
            + f"\n#[cfg(test)]\n#[path={json.dumps(str(test_source))}]\nmod {module};\n"
        )
    harness.write_text(_manifest(f"path={json.dumps(str(staged))}"))
    return harness, staged / "Cargo.toml"


@pytest.mark.parametrize(
    ("kind", "name", "arguments", "arm_only"),
    [
        pytest.param("bin", "output_reuse", ("all",), False, id="output-reuse"),
        pytest.param("bin", "unused_trailing_inputs", (), False, id="trailing-inputs"),
        pytest.param("test", "compressed_success", (), False, id="compressed-status"),
        pytest.param("test", "original_arm_registers", (), True, id="original-arm-abi"),
        pytest.param("bin", "argument_count", ("63",), False, id="arity-63-control"),
        pytest.param("bin", "argument_count", ("64",), False, id="arity-64"),
        pytest.param("bin", "argument_count", ("65",), False, id="arity-65"),
        pytest.param(
            "bin",
            "argument_count",
            ("64", "--no-compress"),
            False,
            id="arity-64-uncompressed-control",
        ),
        pytest.param("bin", "arm_simd_arguments", (), True, id="arm-real-simd"),
        pytest.param(
            "bin",
            "arm_simd_arguments",
            ("--no-compress",),
            True,
            id="arm-real-simd-uncompressed-control",
        ),
        pytest.param("bin", "real_input_locations", (), False, id="real-inputs"),
        pytest.param("test", "arm_normal_return", (), True, id="arm-normal-return"),
        pytest.param("test", "arm_fast_complex", (), True, id="arm-fast-complex"),
    ],
)
def test_public_symjit_regression(
    symjit_manifests: tuple[Path, Path],
    kind: str,
    name: str,
    arguments: tuple[str, ...],
    arm_only: bool,
) -> None:
    if arm_only and not _ARM:
        pytest.skip("this regression exercises AArch64 machine code")
    harness, _ = symjit_manifests
    if kind == "bin":
        _cargo(harness, "run", "--quiet", "--bin", name, "--", *arguments)
    else:
        output = _cargo(harness, "test", "--test", name, "--", "--nocapture")
        assert "running 1 test" in output, f"Rust test was not exercised:\n{output}"


@pytest.mark.parametrize("module", ["return", "abs"])
def test_private_symjit_lowering_regression(
    symjit_manifests: tuple[Path, Path], module: str
) -> None:
    _, source = symjit_manifests
    output = _cargo(
        source, "test", "--lib", f"pyamplicol_{module}_regression", "--", "--nocapture"
    )
    assert "running 1 test" in output, f"Lowering test was not exercised:\n{output}"
