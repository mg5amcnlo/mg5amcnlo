# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import importlib.util
import io
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "dependencies" / "install_dependencies.py"


def _module():
    spec = importlib.util.spec_from_file_location("dependency_installer", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_source_inventory_is_exact_and_profiling_references_are_optional(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    checkouts = tmp_path / "dependencies" / "checkouts"
    checkouts.mkdir(parents=True)
    monkeypatch.setattr(module, "CHECKOUTS", checkouts)
    payload = module._lock()
    without_references = module._sources(
        payload,
        with_legacy=False,
        with_reference_fft=False,
    )
    with_references = module._sources(
        payload,
        with_legacy=True,
        with_reference_fft=True,
    )

    assert {item.key for item in without_references} == {
        "symjit",
        "symbolica",
        "symbolica-community",
        "symbolica-integrate",
        "gammaloop",
        "ratatui-ffi",
        "ufo-model-loader",
    }
    assert {item.key for item in with_references} == {
        *(item.key for item in without_references),
        "legacy-amplicol",
        "reference-fft",
    }
    assert all(len(item.revision) == 40 for item in with_references)
    assert module._root_path_patches() == {}
    assert module._managed_symjit_checkout() == module.CHECKOUTS / "symjit"
    legacy = next(item for item in with_references if item.key == "legacy-amplicol")
    assert legacy.branch == payload["legacy_amplicol"]["branch"]
    assert legacy.revision == payload["legacy_amplicol"]["revision"]
    reference = next(item for item in with_references if item.key == "reference-fft")
    assert reference.url == (
        "https://github.com/rikkert-frederix/AllGluonsMultipletFFT.git"
    )
    assert reference.branch == "main"
    assert reference.revision == "9c3cb4fb4658200884553bab796e85bd5e7fe7a9"


def test_profiling_references_require_explicit_cli_opt_in() -> None:
    module = _module()
    parser = module._parser()

    defaults = parser.parse_args([])
    assert defaults.candidate is False
    assert defaults.loader_wheel is None
    assert defaults.with_legacy_amplicol is False
    assert defaults.with_reference_fft is False

    selected = parser.parse_args(["--with-legacy-amplicol", "--with-reference-fft"])
    assert selected.with_legacy_amplicol is True
    assert selected.with_reference_fft is True
    help_text = parser.format_help()
    assert "--with-legacy-amplicol" in help_text
    assert "--with-reference-fft" in help_text
    assert "--without-legacy-amplicol" not in help_text
    assert "--without-reference-fft" not in help_text


def test_published_sources_retain_dashboard_and_optional_references() -> None:
    module = _module()
    payload = module._lock()
    sources = module._developer_sources(
        payload, with_legacy=False, with_reference_fft=False
    )
    assert len(sources) == 1
    assert sources[0].key == "ratatui-ffi"
    assert sources[0].url == payload["ratatui"]["ffi_repository"]
    assert sources[0].revision == payload["ratatui"]["ffi_revision"]
    assert {
        source.key
        for source in module._developer_sources(
            payload, with_legacy=True, with_reference_fft=True
        )
    } == {"ratatui-ffi", "legacy-amplicol", "reference-fft"}


@pytest.mark.parametrize("candidate", [False, True])
def test_installer_wires_only_explicit_reference_opt_ins(
    monkeypatch: pytest.MonkeyPatch,
    candidate: bool,
) -> None:
    module = _module()
    selections: list[dict[str, bool]] = []
    profiling_extras: list[bool] = []
    monkeypatch.setattr(module, "_lock", lambda: {})

    def sources(_payload, **selection):
        selections.append(selection)
        return ()

    monkeypatch.setattr(module, "_sources", sources)
    monkeypatch.setattr(module, "_developer_sources", sources)
    for name in (
        "_ensure_just",
        "_configure_sources",
        "_write_cargo_config",
        "_write_candidate_lock",
        "_write_state",
    ):
        monkeypatch.setattr(module, name, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        module,
        "_ensure_venv",
        lambda *_args, **kwargs: profiling_extras.append(kwargs["with_fft_profiling"]),
    )

    arguments = ["--dry-run", "--no-build", *(["--candidate"] if candidate else [])]
    assert module.main(arguments) == 0
    assert module.main([*arguments, "--with-legacy-amplicol"]) == 0
    assert module.main([*arguments, "--with-reference-fft"]) == 0
    assert (
        module.main(
            [
                *arguments,
                "--with-legacy-amplicol",
                "--with-reference-fft",
            ]
        )
        == 0
    )
    assert selections == [
        {"with_legacy": False, "with_reference_fft": False},
        {"with_legacy": True, "with_reference_fft": False},
        {"with_legacy": False, "with_reference_fft": True},
        {"with_legacy": True, "with_reference_fft": True},
    ]
    assert profiling_extras == [False, True, True, True]


def test_ratatui_distribution_and_ffi_source_are_exactly_pinned() -> None:
    module = _module()
    ratatui = module._lock()["ratatui"]

    assert ratatui["distribution"] == "ratatui"
    assert ratatui["version"] == "0.4.2"
    assert ratatui["sdist_sha256"] == (
        "2778b066378f8e1629b4b5e1076f1957c8c439b45cdeaf51970d1949730bb0d5"
    )
    assert ratatui["ffi_revision"] == ("7249c0bd1445c0c6ee76f3f24923eb35a1d931e0")
    ffi_source = next(
        source
        for source in module._sources(
            module._lock(),
            with_legacy=False,
            with_reference_fft=False,
        )
        if source.key == "ratatui-ffi"
    )
    assert ffi_source.url == ratatui["ffi_repository"]
    assert ffi_source.revision == ratatui["ffi_revision"]


def test_ufo_loader_distinguishes_required_and_latest_published_versions() -> None:
    module = _module()
    payload = module._lock()
    loader = payload["ufo_model_loader"]
    assert loader["required_version"] == "1.0.0"
    assert loader["latest_verified_published_version"] == "0.1.8"
    assert loader["published_revision"] == ("f5a9f8d2eaff76b99629fc3143952ca4f2425d55")
    assert loader["wheel_sha256"] == (
        "b253942f0a356eacd5fd68b104cacf11039c2391a9ee963107f674fad54f01ef"
    )
    assert loader["release_status"] == "verified"
    assert loader["source_url"] == "https://github.com/alphal00p/ufo_model_loader.git"
    assert loader["candidate_revision"] == "70ddee6b416f8c8b340e0d087646d77095c5d24b"
    assert "patches" not in payload


def test_legacy_oracle_uses_the_pinned_remote_branch_without_local_patches() -> None:
    module = _module()
    payload = module._lock()
    assert "patches" not in payload
    assert not tuple(
        (module.DEPENDENCIES / "patches" / "legacy-amplicol").glob("*.patch")
    )


def test_venv_reset_bootstraps_with_the_unmoved_base_interpreter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    base_python = tmp_path / "base-python"
    active_venv_python = tmp_path / ".venv" / "bin" / "python"
    monkeypatch.setattr(module.sys, "_base_executable", str(base_python))
    monkeypatch.setattr(module.sys, "executable", str(active_venv_python))

    assert module._venv_bootstrap_python() == base_python


def test_local_source_overrides_replace_managed_dependency_clones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    overrides = {
        "symjit": ROOT / "TMP_FIXED_SYMJIT",
        "spenso": ROOT / "TMP_FIXED_SPENSO/crates/spenso",
        "symbolica-integrate": ROOT / "TMP_FIXED_SPENSO/symbolica-integrate",
    }
    monkeypatch.setattr(module, "_root_path_patches", lambda: overrides)
    payload = module._lock()
    symjit = payload["symjit"]

    assert symjit == {
        **module._release_lock()["symjit"],
        **module._contributor_lock()["symjit"],
    }
    sources = {
        item.key
        for item in module._sources(
            payload,
            with_legacy=False,
            with_reference_fft=False,
        )
    }
    assert not {"symjit", "gammaloop", "symbolica-integrate"} & sources
    overrides = module._root_path_patches()
    assert module._managed_symjit_checkout() == ROOT / "TMP_FIXED_SYMJIT"
    assert overrides["spenso"] == ROOT / "TMP_FIXED_SPENSO/crates/spenso"
    assert (
        overrides["symbolica-integrate"]
        == ROOT / "TMP_FIXED_SPENSO/symbolica-integrate"
    )
    assert "patches" not in payload
    assert not tuple((module.DEPENDENCIES / "patches" / "symjit").rglob("*.patch"))


def test_managed_sources_remain_available_without_explicit_path_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.setattr(module, "_root_path_patches", dict)
    sources = {
        source.key: source
        for source in module._sources(
            module._lock(), with_legacy=False, with_reference_fft=False
        )
    }
    assert (
        sources["symjit"].revision == module._contributor_lock()["symjit"]["revision"]
    )
    assert (
        sources["ufo-model-loader"].revision
        == (module._contributor_lock()["ufo_model_loader"]["candidate_revision"])
    )
    contributor = module._contributor_lock()
    assert sources["gammaloop"].branch == contributor["gammaloop_candidate"]["branch"]
    assert (
        sources["gammaloop"].revision == contributor["gammaloop_candidate"]["revision"]
    )
    assert (
        sources["symbolica-integrate"].revision
        == contributor["symbolica_integrate"]["revision"]
    )


def test_community_wiring_preserves_dependency_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    project = tmp_path / "project"
    checkouts = project / "dependencies/checkouts"
    community = checkouts / "symbolica-community"
    (community / "example_extension").mkdir(parents=True)
    (community / "Cargo.toml").write_text(
        '[package]\nname = "symbolica_community"\nversion = "3.0.0"\n'
        '[features]\ndefault = ["module", "native"]\n'
        'module = ["pyo3/abi3", "symbolica/python_abi3"]\n'
        'native = ["symbolica/default", "idenso/native", "spynso3/native"]\n'
        '[dependencies]\nsymbolica = "3.0"\n'
        "[target.'cfg(not(target_arch = \"wasm32\"))'.dependencies]\n"
        'vakint = { git = "https://github.com/alphal00p/gammaloop", '
        'branch = "main" }\n',
        encoding="utf-8",
    )
    (community / "example_extension/Cargo.toml").write_text(
        '[dependencies]\nsymbolica = { version = "3.0", default-features = false }\n',
        encoding="utf-8",
    )
    symjit = project / "TMP_FIXED_SYMJIT"
    symjit.mkdir()
    (symjit / "Cargo.toml").write_text(
        '[package]\nname = "symjit"\n'
        f'version = "{module._contributor_lock()["symjit"]["candidate_version"]}"\n'
        '[lib]\ncrate-type = ["rlib"]\n',
        encoding="utf-8",
    )
    symbolica = checkouts / "symbolica"
    symbolica.mkdir()
    original = b"unchanged upstream dependency manifest and source\n"
    (symbolica / "Cargo.toml").write_bytes(original)
    (symbolica / "source.rs").write_bytes(original)
    spenso = project / "TMP_FIXED_SPENSO/crates/spenso"
    spenso.mkdir(parents=True)
    (spenso / "Cargo.toml").write_bytes(original)
    (spenso / "source.rs").write_bytes(original)
    overrides = {"symjit": symjit, "spenso": spenso}
    monkeypatch.setattr(module, "CHECKOUTS", checkouts)
    monkeypatch.setattr(module, "_root_path_patches", lambda: overrides)

    module._configure_source_manifests(module.Runner(dry_run=False))

    manifest = tomllib.loads((community / "Cargo.toml").read_text())
    assert manifest["patch"]["crates-io"]["symjit"]["path"] == str(symjit)
    assert manifest["patch"]["crates-io"]["spenso"]["path"] == str(spenso)
    git_patch = manifest["patch"]["https://github.com/symbolica-dev/symbolica"]
    assert git_patch["symbolica"]["path"] == str(symbolica)
    assert "build-dependencies" not in manifest
    assert manifest["features"]["default"] == ["module", "native"]
    dependencies = manifest["dependencies"]
    for dependency in (
        "symbolica",
        "idenso",
        "spynso3",
        "symbolica-integrate",
        "example_extension",
    ):
        assert dependencies[dependency]["default-features"] is False
    assert dependencies["symbolica-integrate"]["features"] == [
        "compressed-step-metadata"
    ]
    assert "vakint" not in dependencies
    assert "mimalloc" not in dependencies
    native = manifest["target"]['cfg(not(target_arch = "wasm32"))']["dependencies"]
    assert native["vakint"]["path"] == str(checkouts / "gammaloop/crates/vakint")
    assert native["vakint"]["features"] == ["symbolica_community_module"]
    assert native["mimalloc"]["package"] == "rustfs-mimalloc"
    assert native["mimalloc"]["version"] == "0.5.4"
    assert native["mimalloc"]["features"] == ["local_dynamic_tls"]
    example = tomllib.loads((community / "example_extension/Cargo.toml").read_text())
    assert example["dependencies"]["symbolica"]["default-features"] is False
    for source in (symbolica, spenso):
        assert (source / "Cargo.toml").read_bytes() == original
        assert (source / "source.rs").read_bytes() == original


def test_symjit_checkout_uses_exact_git_revision_and_accepts_matching_head(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    checkouts = tmp_path / "workspace" / "dependencies" / "checkouts"
    monkeypatch.setattr(module, "CHECKOUTS", checkouts)
    source = module.Source(
        "symjit",
        "https://github.com/siravan/symjit-crate.git",
        "d8abfeeb4db98c13cdcf9dd39cf3e795fd5001a7",
    )
    calls: list[tuple[list[str], Path | None, bool]] = []

    class FakeRunner:
        dry_run = False

        def run(self, command, *, cwd=None, capture=False, **_kwargs):
            rendered = [str(item) for item in command]
            calls.append((rendered, cwd, capture))
            return subprocess.CompletedProcess(
                command,
                0,
                source.revision + "\n" if rendered[1:] == ["rev-parse", "HEAD"] else "",
                "",
            )

    module._checkout(FakeRunner(), source, update=False)
    destination = checkouts / "symjit"
    assert calls == [
        (
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                source.url,
                str(destination),
            ],
            None,
            False,
        ),
        (["git", "checkout", "--detach", source.revision], destination, False),
    ]

    destination.mkdir(parents=True)
    (destination / ".git").mkdir()
    calls.clear()
    module._checkout(FakeRunner(), source, update=False)
    assert calls == [(["git", "rev-parse", "HEAD"], destination, True)]


def test_compact_install_state_records_only_git_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    checkouts = tmp_path / "workspace" / "dependencies" / "checkouts"
    state_path = checkouts.parent / "install-state.json"
    sources = (
        module.Source("symjit", "https://example.invalid/symjit.git", "1" * 40),
        module.Source(
            "legacy-amplicol",
            "https://example.invalid/legacy.git",
            "2" * 40,
            "stable",
        ),
    )
    for source in sources:
        (checkouts / source.key).mkdir(parents=True)
    monkeypatch.setattr(module, "CHECKOUTS", checkouts)
    monkeypatch.setattr(module, "STATE", state_path)
    monkeypatch.setattr(module, "_root_path_patches", lambda: {})

    class FakeRunner:
        dry_run = False

        def run(self, command, *, cwd=None, capture=False, **_kwargs):
            assert command == ["git", "rev-parse", "HEAD"]
            assert cwd is not None and capture is True
            revision = next(item.revision for item in sources if item.key == cwd.name)
            return subprocess.CompletedProcess(command, 0, revision + "\n", "")

    module._write_state(FakeRunner(), sources)

    assert module.json.loads(state_path.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "publishable": False,
        "sources": {
            "symjit": {
                "url": "https://example.invalid/symjit.git",
                "revision": "1" * 40,
            },
            "legacy-amplicol": {
                "url": "https://example.invalid/legacy.git",
                "revision": "2" * 40,
                "branch": "stable",
            },
        },
    }


def test_install_state_includes_selected_local_git_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    state_path = tmp_path / "install-state.json"
    paths = {
        "symjit": tmp_path / "TMP_FIXED_SYMJIT",
        "spenso": tmp_path / "TMP_FIXED_SPENSO" / "crates" / "spenso",
        "symbolica-integrate": tmp_path / "TMP_FIXED_SPENSO" / "symbolica-integrate",
    }
    observed: list[Path] = []

    def git_head(_runner, path):
        observed.append(path)
        return "a" * 40

    monkeypatch.setattr(module, "STATE", state_path)
    monkeypatch.setattr(module, "_root_path_patches", lambda: paths)
    monkeypatch.setattr(module, "_git_head", git_head)
    module._write_state(module.Runner(dry_run=False), ())
    state = module.json.loads(state_path.read_text(encoding="utf-8"))
    assert set(state["sources"]) == {"symjit", "gammaloop", "symbolica-integrate"}
    assert observed == list(paths.values())
    assert all(item["revision"] == "a" * 40 for item in state["sources"].values())
    assert (
        state["sources"]["gammaloop"]["branch"]
        == (module._contributor_lock()["gammaloop_candidate"]["branch"])
    )
    assert state["publishable"] is False


@pytest.mark.parametrize(
    "checkout_name",
    ("symbolica", "symbolica-community", "gammaloop"),
)
def test_non_symjit_checkout_symlink_outside_workspace_is_rejected_before_write(
    checkout_name: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    checkouts = tmp_path / "workspace" / "dependencies" / "checkouts"
    checkouts.mkdir(parents=True)
    outside = tmp_path / f"outside-{checkout_name}"
    outside.mkdir()
    marker = outside / "Cargo.toml"
    marker.write_text("outside checkout must remain unchanged\n", encoding="utf-8")
    (checkouts / checkout_name).symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(module, "CHECKOUTS", checkouts)
    source = module.Source(
        checkout_name,
        "https://example.invalid/source.git",
        "1" * 40,
    )
    calls: list[list[str]] = []

    class FakeRunner:
        dry_run = False

        def run(self, command, **_kwargs):
            calls.append([str(item) for item in command])
            return subprocess.CompletedProcess(command, 0, "", "")

    runner = FakeRunner()
    before = marker.read_bytes()
    operations = (
        lambda: module._checkout(runner, source, update=False),
        lambda: module._configure_source_manifests(runner),
    )

    for operation in operations:
        with pytest.raises(module.SetupError, match="must not be a symbolic link"):
            operation()

    assert calls == []
    assert marker.read_bytes() == before
    assert tuple(outside.iterdir()) == (marker,)


def test_legacy_checkout_clones_the_named_branch_then_pins_its_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.setattr(module, "CHECKOUTS", tmp_path / "checkouts")
    source = module.Source(
        "legacy-amplicol",
        "https://github.com/rikkert-frederix/AmpliCol.git",
        "60443f327c2203cf92625da2bf0969c27e68a4ac",
        "amplicol_with_patches",
    )
    calls: list[tuple[list[str], Path | None]] = []

    class FakeRunner:
        dry_run = False

        def run(self, command, *, cwd=None, **_kwargs):
            calls.append(([str(item) for item in command], cwd))
            return subprocess.CompletedProcess(command, 0, "", "")

    module._checkout(FakeRunner(), source, update=False)

    assert calls == [
        (
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--branch",
                "amplicol_with_patches",
                "--single-branch",
                "--no-checkout",
                "https://github.com/rikkert-frederix/AmpliCol.git",
                str(source.path),
            ],
            None,
        ),
        (
            [
                "git",
                "checkout",
                "--detach",
                "60443f327c2203cf92625da2bf0969c27e68a4ac",
            ],
            source.path,
        ),
    ]


def test_checkout_update_migrates_origin_before_fetching(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    checkouts = tmp_path / "checkouts"
    monkeypatch.setattr(module, "CHECKOUTS", checkouts)
    source = module.Source(
        "reference-fft",
        "https://github.com/rikkert-frederix/AllGluonsMultipletFFT.git",
        "9c3cb4fb4658200884553bab796e85bd5e7fe7a9",
        "main",
    )
    destination = checkouts / source.key
    (destination / ".git").mkdir(parents=True)
    calls: list[tuple[list[str], Path | None, bool]] = []

    class FakeRunner:
        dry_run = False

        def run(self, command, *, cwd=None, capture=False, **_kwargs):
            rendered = [str(item) for item in command]
            calls.append((rendered, cwd, capture))
            stdout = "0" * 40 + "\n" if rendered[1:] == ["rev-parse", "HEAD"] else ""
            return subprocess.CompletedProcess(command, 0, stdout, "")

    module._checkout(FakeRunner(), source, update=True)

    assert calls == [
        (["git", "rev-parse", "HEAD"], destination, True),
        (["git", "remote", "set-url", "origin", source.url], destination, False),
        (["git", "fetch", "origin", "main"], destination, False),
        (["git", "checkout", "--detach", source.revision], destination, False),
    ]


def test_contributor_runtime_requirements_use_the_full_hash_locked_closure() -> None:
    module = _module()
    requirements = module._runtime_requirements_text()
    assert "symbolica==" not in requirements
    assert "ufo-model-loader==" not in requirements
    for requirement in (
        "colorama==0.4.6",
        "numpy==2.4.6",
        "prettytable==3.18.0",
        "progressbar2==4.6.0",
        "python-utils==4.1.0",
        "typing-extensions==4.16.0",
        "wcwidth==0.9.1",
    ):
        assert requirements.count(requirement) == 1
    assert requirements.count("--hash=sha256:") > 20


def test_unpublished_ufo_loader_uses_managed_source_without_ignored_checkout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    payload = module._lock()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "CHECKOUTS", tmp_path / "dependencies/checkouts")
    monkeypatch.setattr(module, "_root_path_patches", dict)
    source = next(
        source
        for source in module._sources(
            payload, with_legacy=False, with_reference_fft=False
        )
        if source.key == "ufo-model-loader"
    )
    assert source.path == tmp_path / "dependencies/checkouts/ufo-model-loader"
    assert source.revision == payload["ufo_model_loader"]["candidate_revision"]
    requirements = module._runtime_requirements_text()
    assert "ufo-model-loader==" not in requirements
    assert "numpy==2.4.6" in requirements


def test_contributor_tools_reuse_project_build_test_and_docs_requirements() -> None:
    module = _module()
    with module.PYPROJECT.open("rb") as stream:
        project_file = module.tomllib.load(stream)
    project = project_file["project"]
    optional = project["optional-dependencies"]
    expected = tuple(
        dict.fromkeys(
            (
                *project_file["build-system"]["requires"],
                *optional["test"],
                *optional["docs"],
            )
        )
    )

    requirements = module._contributor_python_requirements()

    assert requirements == expected
    assert requirements.count("maturin==1.15.0") == 1
    assert "pypdf>=5,<6" in requirements

    profiling_requirements = module._contributor_python_requirements(
        with_fft_profiling=True
    )
    assert profiling_requirements == tuple(
        dict.fromkeys((*expected, *optional["fft-profiling"]))
    )
    assert "matplotlib==3.11.2" in profiling_requirements
    assert "reportlab==4.5.1" in profiling_requirements


def test_candidate_dependency_only_build_installs_symbolica_and_managed_ufo_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    venv = tmp_path / ".venv"
    checkouts = tmp_path / "checkouts"
    wheelhouse = tmp_path / "wheelhouse"
    symbolica_wheels = wheelhouse / "symbolica"
    symbolica_wheels.mkdir(parents=True)
    previous_symbolica = symbolica_wheels / "symbolica-2.2.0-test.whl"
    previous_symbolica.write_text("previous Symbolica wheel", encoding="utf-8")
    wheel = symbolica_wheels / "symbolica-3.0.0-test.whl"
    loader_wheels = wheelhouse / "ufo-model-loader"
    loader_wheels.mkdir()
    previous_loader = loader_wheels / "ufo_model_loader-0.1.7-py3-none-any.whl"
    previous_loader.write_text("previous UFO loader wheel", encoding="utf-8")
    loader_wheel = loader_wheels / "ufo_model_loader-0.1.8-py3-none-any.whl"
    calls: list[tuple[list[str], Path | None, dict[str, str] | None]] = []

    class FakeRunner:
        dry_run = False

        def run(self, command, *, cwd=None, env=None, **_kwargs):
            rendered = [str(item) for item in command]
            calls.append((rendered, cwd, env))
            if rendered[1:4] == ["-m", "maturin", "build"]:
                assert not list(symbolica_wheels.glob("*.whl"))
                wheel.touch()
            elif rendered[1:4] == ["-m", "pip", "wheel"]:
                assert not list(loader_wheels.glob("*.whl"))
                loader_wheel.touch()
            return subprocess.CompletedProcess(rendered, 0, "", "")

    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "TRASH", tmp_path / ".trash")
    monkeypatch.setattr(module, "VENV", venv)
    monkeypatch.setattr(module, "CHECKOUTS", checkouts)
    monkeypatch.setattr(module, "WHEELHOUSE", wheelhouse)
    ratatui_payloads: list[dict[str, object]] = []
    monkeypatch.setattr(
        module,
        "_build_ratatui_wheel",
        lambda _runner, payload: ratatui_payloads.append(payload),
    )

    payload: dict[str, object] = {"symbolica": {"candidate_version": "3.0.0"}}
    module._build_candidate_dependency_wheels(
        FakeRunner(),
        payload,
    )

    python = str(venv / "bin" / "python")
    assert calls[0][0][:4] == [python, "-m", "maturin", "build"]
    assert calls[0][1] == checkouts / "symbolica-community"
    assert calls[1][0] == [
        python,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--no-deps",
        str(wheel),
    ]
    assert calls[2][0] == [
        python,
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        "--wheel-dir",
        str(loader_wheels),
        str(checkouts / "ufo-model-loader"),
    ]
    assert calls[3][0] == [
        python,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--no-deps",
        str(loader_wheel),
    ]
    probe, _, environment = calls[4]
    assert probe[:3] == [python, "-I", "-c"]
    assert probe[-1] == "3.0.0"
    assert "from symbolica import Expression" in probe[3]
    assert (
        "from symbolica.community.tensor import TensorExpression, TensorNetwork"
        in probe[3]
    )
    assert "TensorExpression.simplify_algebra" in probe[3]
    assert "update coherent upstream pins" in probe[3]
    assert "community.idenso" not in probe[3]
    assert environment["SYMBOLICA_HIDE_BANNER"] == "1"
    assert ratatui_payloads == [payload]
    archived = {
        path.name: path.read_text(encoding="utf-8")
        for path in (tmp_path / ".trash").rglob("*.whl")
    }
    assert archived == {
        previous_symbolica.name: "previous Symbolica wheel",
        previous_loader.name: "previous UFO loader wheel",
    }


def test_ratatui_build_uses_verified_sdist_and_pinned_local_ffi(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    venv = tmp_path / ".venv"
    checkouts = tmp_path / "checkouts"
    wheelhouse = tmp_path / "wheelhouse"
    ffi_source = checkouts / "ratatui-ffi"
    ffi_source.mkdir(parents=True)
    (ffi_source / "Cargo.toml").write_text(
        '[package]\nname = "ratatui-ffi"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    sdist = wheelhouse / "ratatui" / "ratatui-0.4.2.tar.gz"
    sdist.parent.mkdir(parents=True)
    sdist.touch()
    calls: list[tuple[list[str], dict[str, str] | None]] = []

    class FakeRunner:
        dry_run = False

        def run(self, command, *, env=None, **_kwargs):
            rendered = [str(item) for item in command]
            calls.append((rendered, env))
            if rendered[1:4] == ["-m", "pip", "wheel"]:
                (wheelhouse / "ratatui" / "ratatui-0.4.2-test.whl").touch()
            return subprocess.CompletedProcess(rendered, 0, "", "")

    payload = {
        "ratatui": {
            "version": "0.4.2",
            "ffi_revision": "7249c0bd1445c0c6ee76f3f24923eb35a1d931e0",
        }
    }
    monkeypatch.setattr(module, "VENV", venv)
    monkeypatch.setattr(module, "CHECKOUTS", checkouts)
    monkeypatch.setattr(module, "WHEELHOUSE", wheelhouse)
    monkeypatch.setattr(
        module,
        "_materialize_ratatui_sdist",
        lambda _runner, _payload: sdist,
    )
    monkeypatch.setattr(module, "_archive_candidate_wheels", lambda *_args: None)

    module._build_ratatui_wheel(FakeRunner(), payload)

    build, build_environment = calls[0]
    assert build == [
        str(venv / "bin" / "python"),
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        "--no-build-isolation",
        "--wheel-dir",
        str(wheelhouse / "ratatui"),
        str(sdist),
    ]
    assert build_environment is not None
    assert Path(build_environment["RATATUI_FFI_SRC"]).name == "ratatui-ffi"
    assert Path(build_environment["RATATUI_FFI_SRC"]) != ffi_source.resolve()
    assert build_environment["RATATUI_FFI_TAG"] == payload["ratatui"]["ffi_revision"]
    assert calls[1][0][1:4] == ["-m", "pip", "install"]
    assert calls[2][0][:3] == [str(venv / "bin" / "python"), "-I", "-c"]
    assert "import ratatui" in calls[2][0][3]
    assert "import ratatui_py" in calls[2][0][3]


def test_ratatui_sdist_materialization_rejects_unpinned_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    wheelhouse = tmp_path / "wheelhouse"
    archive = b"not the pinned distribution"
    payload = {
        "ratatui": {
            "version": "0.4.2",
            "source_url": "https://example.invalid/ratatui-0.4.2.tar.gz",
            "sdist_sha256": hashlib.sha256(b"expected distribution").hexdigest(),
        }
    }
    monkeypatch.setattr(module, "WHEELHOUSE", wheelhouse)
    monkeypatch.setattr(
        module.urllib.request,
        "urlopen",
        lambda _url: io.BytesIO(archive),
    )

    with pytest.raises(module.SetupError, match="sdist digest mismatch"):
        module._materialize_ratatui_sdist(module.Runner(dry_run=False), payload)

    assert not (wheelhouse / "ratatui" / "ratatui-0.4.2.tar.gz").exists()


@pytest.mark.parametrize("explicit_bootstrap", [None, "1"])
@pytest.mark.parametrize("mode", ["candidate", "release"])
@pytest.mark.parametrize("platform_name", ["linux", "darwin"])
def test_project_wheel_uses_selected_mode_without_forcing_asset_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    explicit_bootstrap: str | None,
    mode: str,
    platform_name: str,
) -> None:
    module = _module()
    monkeypatch.setattr(module.sys, "platform", platform_name)
    venv = tmp_path / ".venv"
    artifacts = tmp_path / "artifacts"
    calls: list[tuple[list[str], dict[str, str] | None]] = []

    class FakeRunner:
        dry_run = False

        def run(self, command, *, env=None, **_kwargs):
            rendered = [str(item) for item in command]
            calls.append((rendered, env))
            if rendered[1:4] == ["-m", "build", "--wheel"]:
                artifacts.mkdir(parents=True, exist_ok=True)
                (artifacts / "pyamplicol-test.whl").touch()
            return subprocess.CompletedProcess(rendered, 0, "", "")

    if explicit_bootstrap is None:
        monkeypatch.delenv("PYAMPLICOL_PREPARED_MODEL_BOOTSTRAP", raising=False)
    else:
        monkeypatch.setenv(
            "PYAMPLICOL_PREPARED_MODEL_BOOTSTRAP",
            explicit_bootstrap,
        )
    monkeypatch.setattr(module, "VENV", venv)
    monkeypatch.setattr(module, "ARTIFACTS", artifacts)
    monkeypatch.setattr(module, "RELEASE_ARTIFACTS", artifacts)

    module._build_project_wheel(FakeRunner(), mode=mode)

    assert len(calls) == 2
    build_command, build_environment = calls[0]
    assert build_command[1:4] == ["-m", "build", "--wheel"]
    assert "--no-isolation" in build_command
    assert "--skip-dependency-check" in build_command
    assert ("pyamplicol.host-wheel=true" in build_command) == (platform_name == "linux")
    if platform_name == "linux":
        assert (
            build_command[build_command.index("--config-setting") + 1]
            == "pyamplicol.host-wheel=true"
        )
    assert build_environment is not None
    assert build_environment["PYAMPLICOL_BUILD_MODE"] == mode
    install_command, install_environment = calls[1]
    assert install_command[1:4] == ["-m", "pip", "install"]
    assert install_environment is not None
    if explicit_bootstrap is None:
        assert "PYAMPLICOL_PREPARED_MODEL_BOOTSTRAP" not in build_environment
        assert "PYAMPLICOL_PREPARED_MODEL_BOOTSTRAP" not in install_environment
        assert "PYAMPLICOL_PREPARED_MODEL_BOOTSTRAP" not in os.environ
    else:
        assert (
            build_environment["PYAMPLICOL_PREPARED_MODEL_BOOTSTRAP"]
            == explicit_bootstrap
        )
        assert (
            install_environment["PYAMPLICOL_PREPARED_MODEL_BOOTSTRAP"]
            == explicit_bootstrap
        )
        assert os.environ["PYAMPLICOL_PREPARED_MODEL_BOOTSTRAP"] == explicit_bootstrap


def test_dependency_only_and_no_build_are_mutually_exclusive() -> None:
    module = _module()
    with pytest.raises(SystemExit):
        module._parser().parse_args(["--dependencies-only", "--no-build"])


@pytest.mark.parametrize("candidate", [False, True])
@pytest.mark.parametrize("option", ["--dependencies-only", "--no-build"])
def test_dependency_options_skip_the_project_wheel(
    monkeypatch: pytest.MonkeyPatch,
    candidate: bool,
    option: str,
) -> None:
    module = _module()
    payload: dict[str, object] = {}
    monkeypatch.setattr(module, "_lock", lambda: payload)
    monkeypatch.setattr(module, "_sources", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(module, "_developer_sources", lambda *_args, **_kwargs: ())
    for name in (
        "_ensure_just",
        "_ensure_venv",
        "_configure_sources",
        "_write_cargo_config",
        "_write_candidate_lock",
        "_write_state",
    ):
        monkeypatch.setattr(module, name, lambda *_args, **_kwargs: None)
    calls: list[str] = []
    monkeypatch.setattr(
        module,
        "_build_candidate_dependency_wheels",
        lambda *_args: calls.append("dependencies"),
    )
    monkeypatch.setattr(
        module,
        "_install_published_dependencies",
        lambda *_args, **_kwargs: calls.append("published-dependencies"),
    )
    monkeypatch.setattr(
        module, "_build_ratatui_wheel", lambda *_args: calls.append("ratatui")
    )
    monkeypatch.setattr(
        module, "_build_project_wheel", lambda *_args, **_kwargs: pytest.fail("build")
    )
    monkeypatch.setattr(
        module, "_stage_project_runtime", lambda *_args, **_kwargs: pytest.fail("stage")
    )

    assert (
        module.main(
            [
                "--dry-run",
                option,
                *(["--candidate"] if candidate else []),
            ]
        )
        == 0
    )
    expected = ["dependencies"] if candidate else ["published-dependencies", "ratatui"]
    assert calls == ([] if option == "--no-build" else expected)


@pytest.mark.parametrize("with_loader_wheel", [False, True])
def test_published_dependencies_resolve_then_replace_same_version_candidates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    with_loader_wheel: bool,
) -> None:
    module = _module()
    monkeypatch.setattr(module, "VENV", tmp_path / ".venv")
    wheel = tmp_path / "ufo_model_loader-1.0.0-py3-none-any.whl"
    calls: list[list[str]] = []

    class FakeRunner:
        def run(self, command, **_kwargs):
            calls.append([str(item) for item in command])

    module._install_published_dependencies(
        FakeRunner(), loader_wheel=wheel if with_loader_wheel else None
    )
    requirements = tomllib.loads(module.PYPROJECT.read_text())["project"][
        "dependencies"
    ]
    selected = [str(wheel)] if with_loader_wheel else []
    assert calls[0] == [
        str(module._venv_python()),
        "-m",
        "pip",
        "install",
        "--upgrade",
        *requirements,
        *selected,
    ]
    assert calls[1] == [
        str(module._venv_python()),
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--no-deps",
        "--only-binary=:all:",
        "symbolica==3.0.0",
        *(selected or ["ufo-model-loader==1.0.0"]),
    ]


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("mode", ["release", "candidate"])
def test_installer_routes_project_and_staging_together(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    reuse: bool,
    mode: str,
) -> None:
    module = _module()
    monkeypatch.delenv("PYAMPLICOL_BUILD_MODE", raising=False)
    monkeypatch.setattr(module, "_lock", lambda: {})
    monkeypatch.setattr(module, "ARTIFACTS", tmp_path / "candidate")
    monkeypatch.setattr(module, "RELEASE_ARTIFACTS", tmp_path / "release")
    calls: list[tuple[str, object]] = []
    for name in ("_ensure_just", "_ensure_venv"):
        monkeypatch.setattr(module, name, lambda *_args, **_kwargs: None)
    for name in (
        "_configure_sources",
        "_write_cargo_config",
        "_write_candidate_lock",
        "_write_state",
    ):
        monkeypatch.setattr(
            module,
            name,
            lambda *_args, _name=name, **_kwargs: calls.append((_name, None)),
        )
    monkeypatch.setattr(module, "_sources", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(module, "_developer_sources", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(
        module,
        "_build_candidate_dependency_wheels",
        lambda *_args: calls.append(("candidate-deps", None)),
    )
    monkeypatch.setattr(
        module,
        "_install_published_dependencies",
        lambda *_args, **_kwargs: calls.append(("published-deps", None)),
    )
    monkeypatch.setattr(
        module, "_build_ratatui_wheel", lambda *_args: calls.append(("ratatui", None))
    )
    monkeypatch.setattr(
        module,
        "_build_project_wheel",
        lambda _runner, **kwargs: calls.append(("build", kwargs["mode"])),
    )
    monkeypatch.setattr(
        module,
        "_install_project_wheel",
        lambda _runner, directory: calls.append(("install", directory)),
    )
    monkeypatch.setattr(
        module,
        "_stage_project_runtime",
        lambda _runner, **kwargs: calls.append(("stage", kwargs)),
    )
    arguments = ["--dry-run", *(["--candidate"] if mode == "candidate" else [])]
    directory = tmp_path / mode
    if reuse:
        directory = tmp_path / "existing"
        arguments.extend(("--wheel-directory", str(directory)))
    assert module.main(arguments) == 0
    if mode == "release":
        assert calls[:2] == [("published-deps", None), ("ratatui", None)]
        assert len(calls) == 4  # No candidate Cargo overlays or candidate state.
    else:
        assert ("candidate-deps", None) in calls
    assert calls[-2] == (("install", directory) if reuse else ("build", mode))
    assert calls[-1] == ("stage", {"directory": directory, "mode": mode})


@pytest.mark.parametrize("mode", ["release", "candidate"])
def test_source_runtime_stage_reuses_wheel_in_selected_mode(mode: str) -> None:
    module = _module()
    calls = []

    class FakeRunner:
        def run(self, command, **kwargs):
            calls.append((command, kwargs))

    directory = Path("existing-wheels")
    module._stage_project_runtime(FakeRunner(), directory=directory, mode=mode)
    command, options = calls[0]
    assert command == [
        module._venv_python(),
        ROOT / "tools/developer/prepare_source_runtime.py",
        "--wheel-directory",
        directory,
    ]
    assert options["env"]["PYAMPLICOL_BUILD_MODE"] == mode


def test_candidate_mode_rejects_published_loader_override() -> None:
    module = _module()
    with pytest.raises(SystemExit):
        module.main(["--candidate", "--loader-wheel", "loader.whl"])


def test_toml_section_replacement_is_idempotent() -> None:
    module = _module()
    original = '[package]\nname = "x"\n\n[dependencies]\na = "1"\n'
    once = module._replace_section(original, "dependencies", 'b = "2"')
    twice = module._replace_section(once, "dependencies", 'b = "2"')
    assert once == twice
    assert 'a = "1"' not in once
    assert once.count('b = "2"') == 1


def test_symjit_rlib_manifest_validation_does_not_rewrite_source(
    tmp_path: Path,
) -> None:
    module = _module()
    manifest = tmp_path / "Cargo.toml"
    pristine = (
        b'[package]\nname = "symjit"\nversion = "2.22.0"\n\n'
        b'[lib]\ncrate-type = ["rlib"]\n'
    )
    manifest.write_bytes(pristine)

    module._require_symjit_rlib_manifest(manifest, expected_version="2.22.0")

    assert manifest.read_bytes() == pristine

    manifest.write_text(
        '[package]\nname = "symjit"\nversion = "2.22.0"\n\n'
        '[lib]\ncrate-type = ["rlib", "cdylib"]\n',
        encoding="utf-8",
    )
    with pytest.raises(module.SetupError, match="rlib-only"):
        module._require_symjit_rlib_manifest(manifest)

    manifest.write_text(
        '[package]\nname = "symjit"\nversion = "2.21.0"\n\n'
        '[lib]\ncrate-type = ["rlib"]\n',
        encoding="utf-8",
    )
    with pytest.raises(module.SetupError, match="wrong package version"):
        module._require_symjit_rlib_manifest(
            manifest,
            expected_version="2.22.0",
        )


def test_candidate_community_lock_is_resolved_from_the_upstream_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    community = tmp_path / "checkouts" / "symbolica-community"
    community.mkdir(parents=True)
    lock = community / "Cargo.lock"
    lock.write_text("stale generated lock\n", encoding="utf-8")
    calls: list[list[str]] = []

    class FakeRunner:
        dry_run = False

        def run(self, command, *, cwd=None, capture=False, **_kwargs):
            rendered = [str(item) for item in command]
            calls.append(rendered)
            assert cwd == community
            assert capture is True
            if rendered[:2] == ["git", "show"]:
                return subprocess.CompletedProcess(command, 0, "upstream lock\n", "")
            if "--locked" not in rendered:
                assert lock.read_text(encoding="utf-8") == "upstream lock\n"
                lock.write_text("path-resolved lock\n", encoding="utf-8")
            else:
                assert lock.read_text(encoding="utf-8") == "path-resolved lock\n"
            return subprocess.CompletedProcess(command, 0, "{}", "")

    monkeypatch.setattr(module, "CHECKOUTS", tmp_path / "checkouts")
    monkeypatch.setattr(module, "_configure_source_manifests", lambda _runner: None)

    module._configure_sources(FakeRunner())

    assert calls == [
        ["git", "show", "HEAD:Cargo.lock"],
        ["cargo", "metadata", "--format-version", "1"],
        ["cargo", "metadata", "--locked", "--format-version", "1"],
    ]
    assert lock.read_text(encoding="utf-8") == "path-resolved lock\n"


def test_development_symbolica_lock_is_not_publication_resolved(tmp_path: Path) -> None:
    module = _module()
    lock = tmp_path / "Cargo.lock"
    lock.write_text(
        'version = 4\n\n[[package]]\nname = "symbolica"\nversion = "3.0.0"\n'
        'source = "git+https://github.com/symbolica-dev/symbolica#' + "a" * 40 + '"\n',
        encoding="utf-8",
    )
    with pytest.raises(module.SetupError, match="symbolica has an unexpected source"):
        module._validate_release_cargo_lock(lock)


def test_candidate_lock_is_seeded_without_mutating_canonical_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    project = tmp_path / "project"
    project.mkdir()
    (project / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
    release_lock = project / "Cargo.lock"
    release_lock.write_bytes(b"canonical release lock\n")
    (project / "rust").mkdir()
    cargo_config = project / ".cargo" / "config.toml"
    cargo_config.parent.mkdir()
    cargo_config.write_text("[patch.crates-io]\n", encoding="utf-8")
    candidate_lock = project / "dependencies" / "candidate-Cargo.lock"
    candidate_lock.parent.mkdir()
    calls: list[list[str]] = []

    class FakeRunner:
        dry_run = False

        def run(self, command, *, cwd=None, capture=False, **_kwargs):
            assert cwd is not None and capture is True
            calls.append(list(command))
            staged_lock = Path(cwd) / "Cargo.lock"
            if len(calls) == 1:
                assert staged_lock.read_bytes() == b"canonical release lock\n"
                staged_lock.write_bytes(b"candidate path lock\n")
            else:
                assert staged_lock.read_bytes() == b"candidate path lock\n"
            return subprocess.CompletedProcess(command, 0, "{}", "")

    monkeypatch.setattr(module, "ROOT", project)
    monkeypatch.setattr(module, "CARGO_CONFIG", cargo_config)
    monkeypatch.setattr(module, "CANDIDATE_LOCK", candidate_lock)
    monkeypatch.setattr(module, "_root_path_patches", dict)
    monkeypatch.setattr(
        module,
        "_validate_release_cargo_lock",
        lambda _path: pytest.fail("candidate resolution is not release validation"),
    )
    monkeypatch.setattr(module, "_validate_candidate_cargo_lock", lambda _path: None)
    projected: list[Path] = []
    monkeypatch.setattr(
        module,
        "_rewrite_candidate_requirements",
        lambda root: projected.append(root),
    )

    module._write_candidate_lock(FakeRunner())

    assert calls == [
        ["cargo", "metadata", "--format-version", "1"],
        ["cargo", "metadata", "--locked", "--format-version", "1"],
    ]
    assert release_lock.read_bytes() == b"canonical release lock\n"
    assert candidate_lock.read_bytes() == b"candidate path lock\n"
    assert len(projected) == 1
    assert projected[0].parent != project


def test_candidate_dependency_projection_rewrites_only_the_isolated_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    manifest = tmp_path / "rust" / "crates" / "rusticol-core" / "Cargo.toml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        'symbolica = { version = "=2.1.0", default-features = false }\n'
        'symjit = { version = "=2.22.0", default-features = false }\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module,
        "_lock",
        lambda: {
            "symbolica": {
                "rust_version": "2.1.0",
                "candidate_version": "2.2.0",
            },
            "symjit": {
                "version": "2.22.0",
                "candidate_version": "2.21.0",
            },
        },
    )

    module._rewrite_candidate_requirements(tmp_path)

    projected = manifest.read_text(encoding="utf-8")
    assert 'symbolica = { version = "=2.2.0"' in projected
    assert 'symjit = { version = "=2.21.0"' in projected
