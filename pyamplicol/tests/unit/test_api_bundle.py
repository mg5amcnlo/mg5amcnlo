# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import os
import runpy
import shutil
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from pyamplicol.artifacts.api_bundle import (
    api_bundle_payloads,
    format_validation_points,
)

ROOT = Path(__file__).resolve().parents[2]


def test_api_bundle_has_one_complete_root_layout() -> None:
    payloads = api_bundle_payloads(
        {
            "ddbar_to_zg": (
                (500.0, 0.0, 0.0, 500.0),
                (500.0, 0.0, 0.0, -500.0),
                (Decimal("500.0"), "1.25", 2, -3),
                (500.0, -1.25, -2.0, 3.0),
            )
        }
    )
    assert {payload.path for payload in payloads} == {
        "API/validation_points.dat",
        "API/python/check_standalone.py",
        "API/c/check_standalone.c",
        "API/c/Makefile",
        "API/cpp/check_standalone.cpp",
        "API/cpp/Makefile",
        "API/fortran/check_standalone.f90",
        "API/fortran/Makefile",
        "API/rust/check_standalone.rs",
        "API/rust/Makefile",
        "API/umami/Makefile",
        "API/umami/umami.h",
        "API/umami/umami_provider.h",
        "API/umami/umami.c",
        "API/umami/umami_driver.c",
        "API/umami/umami.exports",
        "API/umami/umami.exports.macos",
        "API/umami/UMAMI_LICENSE",
        "API/umami/README.md",
    }
    python = next(payload for payload in payloads if payload.path.endswith(".py"))
    assert python.executable is True
    point = next(payload for payload in payloads if payload.path.endswith(".dat"))
    assert point.content.startswith(b"RUSTICOL_VALIDATION_POINTS_V1\n")
    assert b"ddbar_to_zg\t4\t" in point.content
    makefiles = {
        payload.path: payload.content.decode("utf-8")
        for payload in payloads
        if payload.path.endswith("Makefile")
    }
    assert all("/.pyamplicol-api-build/" in text for text in makefiles.values())
    assert all(
        'cd "$(ARTIFACT_DIR)"' in text
        for path, text in makefiles.items()
        if path != "API/umami/Makefile"
    )
    assert all("API/cpp/check_standalone" not in text for text in makefiles.values())
    assert "CC ?= cc" in makefiles["API/c/Makefile"]
    assert "CXX ?= c++" in makefiles["API/cpp/Makefile"]
    assert "ifeq ($(origin FC), default)" in makefiles["API/fortran/Makefile"]
    assert "FC ?= gfortran" in makefiles["API/fortran/Makefile"]
    rust_makefile = makefiles["API/rust/Makefile"]
    assert "RUSTC ?= rustc" in rust_makefile
    assert '"$(RUSTICOL_CONFIG_PATH)" --rust-source' in rust_makefile
    assert '"$(RUSTICOL_CONFIG_PATH)" --rustflags' in rust_makefile
    assert '"$(RUSTICOL_CONFIG_PATH)" --cargo-rustflags' in rust_makefile
    assert ".venv/bin/rusticol-config" in rust_makefile
    assert "rusticol-config is unavailable" in rust_makefile
    assert "RUSTICOL_RUST_SOURCE" in rust_makefile
    assert "CARGO_ENCODED_RUSTFLAGS" in rust_makefile
    assert "run-script:" in rust_makefile
    assert "TARGET := $(BUILD_DIR)/check_standalone" in rust_makefile
    assert "$(RUSTC) --edition=2021" in rust_makefile
    assert "\tcargo " not in rust_makefile.lower()
    assert "Cargo.toml" not in rust_makefile

    rust_source = next(
        payload.content.decode("utf-8")
        for payload in payloads
        if payload.path == "API/rust/check_standalone.rs"
    )
    assert "#!/usr/bin/env rust-script" in rust_source
    assert 'include!(env!("RUSTICOL_RUST_SOURCE"))' in rust_source
    assert "Runtime::load" in rust_source
    assert "Selectors::all()" in rust_source
    assert ".set_model_parameters(&options.overrides)" in rust_source
    assert "--kinematics" in rust_source
    assert ".representative_process_key()" in rust_source
    assert ".external_permutation()" in rust_source
    assert ".load_kinematics_json(" in rust_source
    assert ".evaluate_f64(&momenta, 1)" in rust_source
    assert ".evaluate_resolved_f64(&momenta, 1" in rust_source
    assert r"\"language\":\"rust\"" in rust_source
    assert "unsafe" not in rust_source
    assert 'extern "C"' not in rust_source
    assert "rusticol_runtime_" not in rust_source
    assert "extern crate" not in rust_source
    assert "serde" not in rust_source

    umami_makefile = makefiles["API/umami/Makefile"]
    assert "include providers.mk" in umami_makefile
    assert "libumami.so" in umami_makefile
    assert '"$(RUSTICOL_CONFIG_PATH)" --cflags' in umami_makefile
    assert '"$(RUSTICOL_CONFIG_PATH)" --libs' in umami_makefile
    # UMAMI receives the absolute artifact path explicitly. Keep the caller's
    # working directory so an overridden relative BUILD_DIR remains valid.
    assert '\t"$(RUN_DRIVER)" "$(ARTIFACT_DIR)" $(ARGS)' in umami_makefile
    assert 'cd "$(ARTIFACT_DIR)"' not in umami_makefile

    c_source = next(
        payload.content.decode("utf-8")
        for payload in payloads
        if payload.path == "API/c/check_standalone.c"
    )
    assert "#include <rusticol.h>" in c_source
    assert "rusticol_runtime_load" in c_source
    assert "rusticol_runtime_evaluate_f64" in c_source
    assert "rusticol_runtime_evaluate_resolved_f64" in c_source
    assert "rusticol_runtime_representative_process_key" in c_source
    assert "rusticol_runtime_external_permutation" in c_source
    assert "rusticol_runtime_load_kinematics_json" in c_source

    # Emitted drivers import the installed SDK instead of duplicating its
    # warm-up ABI; updating construction options must not change lazy defaults.
    cpp_source = next(
        payload.content.decode("utf-8")
        for payload in payloads
        if payload.path == "API/cpp/check_standalone.cpp"
    )
    fortran_source = next(
        payload.content.decode("utf-8")
        for payload in payloads
        if payload.path == "API/fortran/check_standalone.f90"
    )
    assert "#include <rusticol.hpp>" in cpp_source
    assert "use rusticol" in fortran_source
    for source in (c_source, cpp_source, fortran_source, rust_source):
        assert "warm_up(" not in source
        assert "warm_up_f64(" not in source

    for payload in payloads:
        if "check_standalone" in payload.path:
            assert b"--kinematics" in payload.content


def _write_api_bundle(root: Path) -> Path:
    api = root / "artifact/API"
    for payload in api_bundle_payloads():
        target = root / "artifact" / payload.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload.content)
        if payload.executable:
            target.chmod(0o755)
    (api / "umami/providers.mk").write_text(
        "UMAMI_PROVIDERS := p0\nUMAMI_SINGLE_PROVIDER := 1\n", encoding="ascii"
    )
    (api / "umami/provider_p0.h").write_text(
        "/* dry-run fixture; real providers are generated per artifact */\n",
        encoding="ascii",
    )
    return api


def _write_sdk_config(path: Path, marker: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    library = path.parent / "lib-fixture.a"
    library.write_bytes(b"")
    path.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        f"  --cflags) printf '%s\\n' '-I/{marker}/include' ;;\n"
        f"  --libs) printf '%s\\n' '/{marker}/lib.a' ;;\n"
        f"  --library) printf '%s\\n' '{library}' ;;\n"
        f"  --fortran-source) printf '%s\\n' '/{marker}/rusticol.f90' ;;\n"
        f"  --rust-source) printf '%s\\n' '/{marker}/rusticol.rs' ;;\n"
        f"  --rustflags) printf '%s\\n' '-L /{marker}' ;;\n"
        f"  --cargo-rustflags) printf '%s\\n' '-L{marker}' ;;\n"
        "  *) exit 2 ;;\n"
        "esac\n",
        encoding="ascii",
    )
    path.chmod(0o755)


def test_api_makefiles_find_source_checkout_sdk_without_activated_path(
    tmp_path: Path,
) -> None:
    make = shutil.which("make")
    if make is None:
        pytest.skip("make is unavailable")
    checkout = tmp_path / "checkout"
    api = _write_api_bundle(checkout / "examples")
    _write_sdk_config(checkout / ".venv/bin/rusticol-config", "ancestor-sdk")
    environment = os.environ.copy()
    environment["PATH"] = "/usr/bin:/bin"
    environment.pop("RUSTICOL_CONFIG", None)

    outputs: dict[str, str] = {}
    for language in ("c", "cpp", "fortran", "rust", "umami"):
        completed = subprocess.run(
            [make, "-n", "-C", str(api / language)],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=20,
        )
        assert completed.returncode == 0, completed.stderr
        outputs[language] = completed.stdout
    assert "/ancestor-sdk/include" in outputs["c"]
    assert "/ancestor-sdk/include" in outputs["cpp"]
    assert "/ancestor-sdk/rusticol.f90" in outputs["fortran"]
    assert "/ancestor-sdk/rusticol.rs" in outputs["rust"]
    assert "/ancestor-sdk/include" in outputs["umami"]
    assert "libumami.so" in outputs["umami"]

    override = tmp_path / "override-rusticol-config"
    _write_sdk_config(override, "explicit-sdk")
    completed = subprocess.run(
        [make, "-n", "-C", str(api / "c"), f"RUSTICOL_CONFIG={override}"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    assert "/explicit-sdk/include" in completed.stdout
    assert "/ancestor-sdk/include" not in completed.stdout

    isolated_api = _write_api_bundle(tmp_path / "isolated")
    missing = subprocess.run(
        [make, "-n", "-C", str(isolated_api / "c")],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=20,
    )
    assert missing.returncode != 0
    assert "rusticol-config is unavailable" in missing.stderr
    assert "rusticol.h" not in missing.stderr


def test_python_api_driver_reexecutes_nearest_source_checkout_python(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    api = _write_api_bundle(checkout / "examples")
    record = checkout / "reexec-argv.txt"
    python = checkout / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$@\" > " + repr(os.fspath(record)) + "\n",
        encoding="ascii",
    )
    python.chmod(0o755)
    _write_sdk_config(checkout / ".venv/bin/rusticol-config", "ancestor-sdk")
    environment = os.environ.copy()
    environment.pop("_PYAMPLICOL_API_BOOTSTRAP", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(api / "python/check_standalone.py"),
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    assert record.read_text(encoding="utf-8").splitlines() == [
        str((api / "python/check_standalone.py").resolve()),
        "--help",
    ]


def test_validation_points_are_sorted_and_require_four_vectors() -> None:
    output = format_validation_points(
        {
            "second": ((1, 2, 3, 4),),
            "first": ((5, 6, 7, 8),),
        }
    ).decode("ascii")
    assert output.splitlines()[1].startswith("first\t")
    assert output.splitlines()[2].startswith("second\t")
    with pytest.raises(ValueError, match="four-vectors"):
        format_validation_points({"broken": ((1, 2, 3),)})
    with pytest.raises(ValueError, match="non-empty tokens"):
        format_validation_points({"not valid": ((1, 2, 3, 4),)})


def test_python_api_driver_enforces_canonical_model_parameter_card_values() -> None:
    namespace = runpy.run_path(
        str(ROOT / "src/pyamplicol/assets/api_templates/python/check_standalone.py")
    )
    convert = namespace["_parameter_value"]

    assert convert(0.117) == 0.117
    assert convert([2.5, -0.25]) == complex(2.5, -0.25)
    for invalid in (True, "0.117", [True, 0], [0, False], [1], {}, float("inf")):
        with pytest.raises(SystemExit):
            convert(invalid)
