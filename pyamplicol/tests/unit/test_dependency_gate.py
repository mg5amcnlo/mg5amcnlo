# SPDX-License-Identifier: 0BSD

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "release" / "check_dependencies.py"


def _module():
    spec = importlib.util.spec_from_file_location("dependency_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_release_gate_accepts_canonical_published_cargo_inputs() -> None:
    module = _module()
    assert module.check(candidate=False) == []


@pytest.fixture
def published_cargo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    module = _module()
    release = module._load_lock()
    symjit = release["symjit"]
    symbolica = release["symbolica"]
    (tmp_path / "Cargo.toml").write_text(
        "[workspace.dependencies]\n"
        f'symbolica = {{ version = "={symbolica["rust_version"]}" }}\n',
        encoding="utf-8",
    )
    core = tmp_path / "rust/crates/rusticol-core/Cargo.toml"
    core.parent.mkdir(parents=True)
    core.write_text(
        "[dependencies]\n"
        f'symbolica = {{ version = "={symbolica["rust_version"]}" }}\n'
        f'symjit = {{ version = "={symjit["version"]}" }}\n',
        encoding="utf-8",
    )
    cargo_lock = tmp_path / "Cargo.lock"
    cargo_lock.write_text(
        'version = 4\n[[package]]\nname = "symbolica"\n'
        f'version = "{symbolica["rust_version"]}"\n'
        f'source = "{module._REGISTRY_SOURCE}"\nchecksum = "{"0" * 64}"\n'
        '[[package]]\nname = "symjit"\n'
        f'version = "{symjit["version"]}"\n'
        f'source = "{module._REGISTRY_SOURCE}"\nchecksum = "{"1" * 64}"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "CARGO_LOCK_PATH", cargo_lock)
    return module


def test_release_contract_is_lean_exact_and_schema_8() -> None:
    module = _module()
    lock = module._load_lock()
    assert lock["abis"]["compiled_model"] == 9
    assert "python_runtime_lock" not in lock
    assert "legal_status" not in lock
    assert module._locked_python_dependencies(lock) == (
        module._project_python_dependencies()
    )
    assert set(lock["symbolica"]) == {
        "python_distribution",
        "python_version",
        "rust_crate",
        "rust_version",
        "serialization_abi",
    }
    assert lock["symjit"] == {"version": "2.26.4"}
    assert set(lock["ufo_model_loader"]) == {
        "python_distribution",
        "required_version",
    }


def test_release_cargo_lock_accepts_registry_crates_including_symjit(
    published_cargo,
) -> None:
    assert (
        published_cargo._release_cargo_lock_issues(published_cargo._load_lock()) == []
    )


def test_release_toolchain_and_manylinux_image_are_exactly_pinned() -> None:
    module = _module()
    assert module._toolchain_issues(module._load_lock()) == []


def test_release_cargo_lock_rejects_candidate_path_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    published_cargo,
) -> None:
    module = published_cargo
    contaminated = tmp_path / "Cargo.lock"
    text = module.CARGO_LOCK_PATH.read_text(encoding="utf-8")
    marker = (
        'name = "symbolica"\n'
        f'version = "{module._load_lock()["symbolica"]["rust_version"]}"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
    )
    assert marker in text
    contaminated.write_text(text.replace(marker, marker.rsplit("source", 1)[0], 1))
    monkeypatch.setattr(module, "CARGO_LOCK_PATH", contaminated)
    codes = {
        issue.code for issue in module._release_cargo_lock_issues(module._load_lock())
    }
    assert codes == {
        "release-cargo-nonregistry",
        "release-cargo-pin",
    }


def test_candidate_gate_fails_closed_before_contributor_install(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    monkeypatch.setattr(module, "STATE_PATH", tmp_path / "install-state.json")
    monkeypatch.setattr(module, "CANDIDATE_LOCK_PATH", tmp_path / "Cargo.lock")
    monkeypatch.setattr(module, "CARGO_CONFIG_PATH", tmp_path / "config.toml")
    codes = {issue.code for issue in module.check(candidate=True)}
    assert "candidate-input-missing" in codes
    assert "symbolica-unverified" not in codes


def test_candidate_gate_uses_candidate_not_publication_cargo_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    publication_issue = module.GateIssue("release-cargo-patch", "local override")
    monkeypatch.setattr(module, "_release_contract_issues", lambda lock: [])
    monkeypatch.setattr(module, "_toolchain_issues", lambda lock: [])
    monkeypatch.setattr(module, "_candidate_issues", lambda lock: [])
    monkeypatch.setattr(
        module, "_release_cargo_lock_issues", lambda lock: [publication_issue]
    )

    assert module.check(candidate=True) == []
    assert module.check(candidate=False) == [publication_issue]


@pytest.mark.parametrize("local_overrides", (False, True))
def test_candidate_gate_uses_compact_exact_git_sources_and_rlib_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    local_overrides: bool,
) -> None:
    module = _module()
    release = module._load_lock()
    contributor = module._load_contributor_lock()
    expected = module._candidate_sources(contributor, release)
    dependencies = tmp_path / "dependencies"
    contributor_path = dependencies / "contributor-lock.toml"
    state_path = dependencies / "install-state.json"
    candidate_lock = dependencies / "candidate-Cargo.lock"
    cargo_config = dependencies / "candidate-cargo-config.toml"
    checkouts = dependencies / "checkouts"
    contributor_path.parent.mkdir(parents=True)
    contributor_path.write_bytes(module.CONTRIBUTOR_LOCK_PATH.read_bytes())
    source_paths = {name: checkouts / name for name in expected}
    if local_overrides:
        source_paths.update(
            {
                "symjit": tmp_path / "local-symjit",
                "gammaloop": tmp_path / "local-gammaloop",
                "symbolica-integrate": tmp_path / "local-integrate",
            }
        )
    for path in source_paths.values():
        path.mkdir(parents=True)
    for name in ("graphica", "numerica"):
        (checkouts / "symbolica" / "lib" / name).mkdir(parents=True)
    (source_paths["symjit"] / "Cargo.toml").write_text(
        '[package]\nname = "symjit"\n'
        f'version = "{contributor["symjit"]["candidate_version"]}"\n\n'
        '[lib]\ncrate-type = ["rlib"]\n',
        encoding="utf-8",
    )
    compact_state = {
        "schema_version": 1,
        "publishable": False,
        "sources": {
            name: {"url": url, "revision": revision}
            for name, (url, revision) in expected.items()
            if source_paths[name] == checkouts / name
        },
    }
    state_path.write_text(json.dumps(compact_state), encoding="utf-8")
    candidate_lock.write_text(
        'version = 4\n\n[[package]]\nname = "rusticol-core"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    patches = {
        "graphica": checkouts / "symbolica/lib/graphica",
        "numerica": checkouts / "symbolica/lib/numerica",
        "symbolica": checkouts / "symbolica",
        "symjit": source_paths["symjit"],
    }
    overrides = {}
    if local_overrides:
        overrides = {
            "symjit": source_paths["symjit"],
            "spenso": source_paths["gammaloop"] / "crates/spenso",
            "symbolica-integrate": source_paths["symbolica-integrate"],
        }
        overrides["spenso"].mkdir(parents=True)
        patches.update(overrides)
    (tmp_path / "Cargo.toml").write_text(
        "[patch.crates-io]\n"
        + "".join(
            f'{name} = {{ path = "{path}" }}\n' for name, path in overrides.items()
        ),
        encoding="utf-8",
    )
    config_text = "[patch.crates-io]\n" + "".join(
        f'{name} = {{ path = "{path}" }}\n' for name, path in patches.items()
    )
    cargo_config.write_text(config_text, encoding="utf-8")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "CONTRIBUTOR_LOCK_PATH", contributor_path)
    monkeypatch.setattr(module, "STATE_PATH", state_path)
    monkeypatch.setattr(module, "CANDIDATE_LOCK_PATH", candidate_lock)
    monkeypatch.setattr(module, "CARGO_CONFIG_PATH", cargo_config)
    monkeypatch.setattr(module, "CHECKOUTS_PATH", checkouts)
    revisions = {
        source_paths[name]: revision for name, (_, revision) in expected.items()
    }
    monkeypatch.setattr(module, "_git_head", revisions.get)

    assert module._candidate_issues(release) == []

    monkeypatch.setattr(module, "_git_head", lambda path: "0" * 40)
    assert "candidate-source-revision" in {
        issue.code for issue in module._candidate_issues(release)
    }
    monkeypatch.setattr(module, "_git_head", revisions.get)

    cargo_config.write_text(
        config_text.replace(str(source_paths["symjit"]), str(tmp_path / "undeclared")),
        encoding="utf-8",
    )
    assert "candidate-cargo-config" in {
        issue.code for issue in module._candidate_issues(release)
    }
    cargo_config.write_text(config_text, encoding="utf-8")

    heavy_state = copy.deepcopy(compact_state)
    heavy_state["sources"]["symbolica"]["worktree_sha256"] = "0" * 64
    state_path.write_text(json.dumps(heavy_state), encoding="utf-8")
    assert "candidate-state-invalid" in {
        issue.code for issue in module._candidate_issues(release)
    }

    state_path.write_text(json.dumps(compact_state), encoding="utf-8")
    patch_root = dependencies / "patches" / "symjit"
    patch_root.mkdir(parents=True)
    (patch_root / "legacy.patch").write_text("obsolete\n", encoding="utf-8")
    assert "candidate-symjit-patch" in {
        issue.code for issue in module._candidate_issues(release)
    }

    (source_paths["symjit"] / "Cargo.toml").write_text(
        '[package]\nname = "symjit"\n'
        f'version = "{contributor["symjit"]["candidate_version"]}"\n\n'
        '[lib]\ncrate-type = ["rlib", "cdylib"]\n',
        encoding="utf-8",
    )
    assert "candidate-symjit-manifest" in {
        issue.code for issue in module._candidate_issues(release)
    }


def test_release_contract_rejects_symjit_git_source_metadata() -> None:
    module = _module()
    release = copy.deepcopy(module._load_lock())
    release["symjit"]["repository"] = (
        "https://github.com/ValentinHirschi/symjit_crate_changes_for_pyamplicol.git"
    )

    assert "release-lock-scope" in {
        issue.code for issue in module._release_contract_issues(release)
    }


def test_release_contract_pins_symjit_plane_application_abi() -> None:
    module = _module()
    release = module._load_lock()

    assert not {
        issue.code
        for issue in module._release_contract_issues(release)
        if issue.code == "release-abi-contract"
    }

    missing = copy.deepcopy(release)
    del missing["abis"]["symjit_plane_application"]
    assert "release-abi-contract" in {
        issue.code for issue in module._release_contract_issues(missing)
    }

    wrong = copy.deepcopy(release)
    wrong["abis"]["symjit_plane_application"] = "wrong"
    assert "release-abi-contract" in {
        issue.code for issue in module._release_contract_issues(wrong)
    }


def test_release_gate_has_no_live_package_or_repository_preflight() -> None:
    module = _module()
    source = SCRIPT.read_text(encoding="utf-8")
    assert not hasattr(module, "_published_dependency_issues")
    assert "urllib.request" not in source
    assert "pypi.org/pypi/" not in source
    assert "crates.io/api/" not in source
