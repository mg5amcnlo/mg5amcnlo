# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import sys
import tomllib

import pytest

from tools.typing import check_public_typing as typing_gate
from tools.typing.check_public_typing import (
    PUBLIC_FACADES,
    ROOT,
    MetadataCheck,
    check_typing_metadata,
    render_export_consumer,
    static_all,
)


def test_main_propagates_source_mypy_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def successful_metadata_check() -> MetadataCheck:
        return MetadataCheck(errors=(), native_symbols=())

    def failed_mypy(
        command: list[str], *, environment: dict[str, str] | None = None
    ) -> int:
        assert command == [sys.executable, "-m", "mypy"]
        assert environment is None
        return 2

    def unexpected_installed_check() -> int:
        raise AssertionError("installed typing checks must not mask source failures")

    monkeypatch.setattr(typing_gate, "check_typing_metadata", successful_metadata_check)
    monkeypatch.setattr(typing_gate, "_run_mypy", failed_mypy)
    monkeypatch.setattr(
        typing_gate, "_check_installed_consumers", unexpected_installed_check
    )

    assert typing_gate.main() == 2


def test_mypy_scope_is_explicit_and_has_no_blanket_public_ignores() -> None:
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    mypy = configuration["tool"]["mypy"]

    assert mypy["strict"] is True
    assert mypy["follow_imports"] == "silent"
    assert "packages" not in mypy
    assert "src/pyamplicol" not in mypy["files"]
    assert "src/pyamplicol/__init__.py" in mypy["files"]
    assert "src/pyamplicol/api" in mypy["files"]
    assert "src/pyamplicol/config/models.py" in mypy["files"]
    assert "src/pyamplicol/assets/models/ufo/" in "\n".join(mypy["exclude"])
    assert "src/pyamplicol/assets/api_templates/" in "\n".join(mypy["exclude"])
    assert all(
        not override.get("ignore_errors", False)
        for override in configuration["tool"].get("mypy", {}).get("overrides", [])
    )


def test_public_facade_exports_are_static_and_in_the_generated_consumer() -> None:
    consumer = render_export_consumer()
    for module, path in PUBLIC_FACADES.items():
        exports = static_all(path)
        assert exports
        alias = module.rsplit(".", 1)[-1] if "." in module else "package"
        assert all(f"reveal_type({alias}.{name})" in consumer for name in exports)


def test_typing_marker_and_native_stub_cover_the_packaged_contract() -> None:
    result = check_typing_metadata()
    assert result.errors == ()
    assert (ROOT / "src" / "pyamplicol" / "py.typed").is_file()


def test_consumers_are_outside_the_package_tree() -> None:
    package = ROOT / "src" / "pyamplicol"
    for name in ("consumer_public_api.py", "consumer_native_api.py"):
        consumer = ROOT / "tests" / "typing" / name
        assert consumer.is_file()
        assert package not in consumer.parents
