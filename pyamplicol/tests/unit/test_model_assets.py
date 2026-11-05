# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import json
import math
import re
import tomllib
from importlib import resources
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ASSET_ROOT = PROJECT_ROOT / "src" / "pyamplicol" / "assets" / "models"
MANIFEST_LINE = re.compile(r"([0-9a-f]{64})  ([^\r\n]+)")
HOST_PATH_MARKERS = (b"/Users/", b"/home/", b"file://")

EXPECTED_UFO_FILES = {
    "scalar_gravity": {
        "__init__.py",
        "coupling_orders.py",
        "couplings.py",
        "function_library.py",
        "lorentz.py",
        "object_library.py",
        "param_card.dat",
        "parameters.py",
        "particles.py",
        "restrict_default.dat",
        "restrict_full.dat",
        "run_write_param_card.py",
        "vertices.py",
        "write_param_card.py",
    },
    "scalars": {
        "__init__.py",
        "coupling_orders.py",
        "couplings.py",
        "function_library.py",
        "lorentz.py",
        "object_library.py",
        "parameters.py",
        "particles.py",
        "restrict_default.dat",
        "restrict_full.dat",
        "run_write_param_card.py",
        "vertices.py",
        "write_param_card.py",
    },
    "sm": {
        "__init__.py",
        "build_restrict.py",
        "coupling_orders.py",
        "couplings.py",
        "decays.py",
        "function_library.py",
        "lorentz.py",
        "object_library.py",
        "parameters.py",
        "particles.py",
        "restrict_c_mass.dat",
        "restrict_ckm.dat",
        "restrict_default.dat",
        "restrict_full.dat",
        "restrict_lepton_masses.dat",
        "restrict_no_b_mass.dat",
        "restrict_no_masses.dat",
        "restrict_no_tau_mass.dat",
        "restrict_no_widths.dat",
        "restrict_zeromass_ckm.dat",
        "run_write_param_card.py",
        "vertices.py",
        "write_param_card.py",
    },
}

EXPECTED_MODEL_SHAPES = {
    "scalar_gravity/scalar_gravity.json": (4, 11, 14, 8),
    "scalars/scalars.json": (3, 276, 1, 8),
    "scalars/scalars_2p_3p.json": (3, 16, 1, 2),
    "sm/sm.json": (43, 153, 108, 22),
    "sm/sm_wrapped_indices.json": (43, 153, 108, 22),
}


def _assets_resource():
    return resources.files("pyamplicol").joinpath("assets", "models")


def _resource(relative: str):
    current = _assets_resource()
    for part in PurePosixPath(relative).parts:
        current = current.joinpath(part)
    return current


def _is_interpreter_cache(path: Path, *, root: Path) -> bool:
    relative = path.relative_to(root)
    return "__pycache__" in relative.parts and path.suffix == ".pyc"


def _manifest() -> tuple[list[str], dict[str, str]]:
    lines = _resource("MANIFEST.sha256").read_text(encoding="utf-8").splitlines()
    assert lines
    entries: dict[str, str] = {}
    paths: list[str] = []
    for line in lines:
        match = MANIFEST_LINE.fullmatch(line)
        assert match is not None, f"malformed manifest line: {line!r}"
        digest, relative = match.groups()
        path = PurePosixPath(relative)
        assert not path.is_absolute()
        assert relative == path.as_posix()
        assert "\\" not in relative
        assert not {"", ".", ".."}.intersection(path.parts)
        assert relative not in entries
        entries[relative] = digest
        paths.append(relative)
    return paths, entries


def test_manifest_is_exact_and_resources_are_readable() -> None:
    paths, manifest = _manifest()
    assert paths == sorted(paths)
    assert "MANIFEST.sha256" not in manifest
    assert "PROVENANCE.toml" in manifest

    asset_root = Path(str(_assets_resource()))
    actual = {
        path.relative_to(asset_root).as_posix()
        for path in asset_root.rglob("*")
        if path.is_file()
        and path.name != "MANIFEST.sha256"
        and not _is_interpreter_cache(path, root=asset_root)
    }
    assert set(manifest) == actual

    for relative, expected in manifest.items():
        resource = _resource(relative)
        assert resource.is_file(), relative
        assert hashlib.sha256(resource.read_bytes()).hexdigest() == expected


def test_assets_have_no_links_unsafe_paths_or_generated_files() -> None:
    asset_root = Path(str(_assets_resource()))
    resolved_root = asset_root.resolve()
    installed_copy = resolved_root != SOURCE_ASSET_ROOT.resolve()
    for path in asset_root.rglob("*"):
        assert not path.is_symlink(), path
        assert path.resolve().is_relative_to(resolved_root), path
        relative = path.relative_to(asset_root)
        if path.name == "__pycache__" or _is_interpreter_cache(path, root=asset_root):
            assert installed_copy, path
            assert path.name == "__pycache__" or path.suffix == ".pyc", path
            continue
        assert "__pycache__" not in relative.parts
        assert path.suffix != ".pyc"
        if not path.is_file():
            continue
        content = path.read_bytes()
        assert b"../" not in content, path
        assert b"..\\" not in content, path
        assert not any(marker in content for marker in HOST_PATH_MARKERS), path


def test_provenance_covers_every_source_asset_and_license() -> None:
    _, manifest = _manifest()
    provenance = tomllib.loads(_resource("PROVENANCE.toml").read_text(encoding="utf-8"))
    assert provenance["schema_version"] == 1
    assert provenance["release_status"] == "eligible"
    assert provenance["source"]["revision"] == (
        "643bc6f99d7b2249af0a85204768df243e612411"
    )
    assert provenance["generator"]["version"] == "0.1.8"
    assert re.fullmatch(r"[0-9a-f]{40}", provenance["generator"]["revision"])

    licenses = {entry["id"]: entry for entry in provenance["licenses"]}
    assert set(licenses) == {
        "0BSD",
        "GammaLoop-model-assets",
        "MIT",
        "MadGraph5-aMCatNLO",
    }
    for entry in licenses.values():
        assert (PROJECT_ROOT / entry["path"]).is_file(), entry["path"]

    transformations = {entry["id"] for entry in provenance["transformations"]}
    records = provenance["files"]
    assert len(records) == 66
    assert sum(entry["upstream_exact_blob_match"] for entry in records) == 27

    record_paths = set()
    for entry in records:
        relative = entry["canonical_package_path"].removeprefix(
            "pyamplicol/assets/models/"
        )
        assert relative != entry["canonical_package_path"]
        assert relative in manifest
        assert entry["package_sha256"] == manifest[relative]
        assert re.fullmatch(r"[0-9a-f]{64}", entry["source_sha256"])
        assert re.fullmatch(r"[0-9a-f]{40}", entry["source_blob_oid"])
        assert entry["transformation_id"] in transformations
        assert set(entry["license_ids"]).issubset(licenses)
        record_paths.add(relative)

    assert record_paths == set(manifest) - {"PROVENANCE.toml"}
    models = {entry["id"]: entry for entry in provenance["models"]}
    assert models["built-in-sm"]["canonical_package_path"] == (
        "pyamplicol/models/builtin/__init__.py"
    )
    assert models["built-in-sm"]["source_sha256"] == (
        "7b3bac9e53193260bf9c2829ef972f87d23246e93dc57f208bda4821195b7ad3"
    )
    assert all(entry["release_eligible"] for entry in models.values())


def test_ufo_directories_are_complete_and_syntactically_valid() -> None:
    ufo_root = Path(str(_assets_resource().joinpath("ufo")))
    assert {path.name for path in ufo_root.iterdir() if path.is_dir()} == set(
        EXPECTED_UFO_FILES
    )
    for model, expected in EXPECTED_UFO_FILES.items():
        model_root = ufo_root / model
        actual = {path.name for path in model_root.iterdir() if path.is_file()}
        assert actual == expected
        for path in model_root.glob("*.py"):
            compile(path.read_bytes(), path.as_posix(), "exec")


def test_all_json_is_parseable_and_models_have_expected_shape() -> None:
    json_root = Path(str(_assets_resource().joinpath("json")))
    json_files = sorted(json_root.rglob("*.json"))
    assert len(json_files) == 16
    payloads = {
        path.relative_to(json_root).as_posix(): json.loads(
            path.read_text(encoding="utf-8")
        )
        for path in json_files
    }
    assert all(isinstance(payload, dict) for payload in payloads.values())

    for relative, expected in EXPECTED_MODEL_SHAPES.items():
        payload = payloads[relative]
        actual = tuple(
            len(payload[key])
            for key in (
                "particles",
                "vertex_rules",
                "couplings",
                "lorentz_structures",
            )
        )
        assert actual == expected


def test_json_restrictions_are_complete_explicit_ufo_cards() -> None:
    from ufo_model_loader.common import optionally_lower_external_parameter_name
    from ufo_model_loader.model import InputParamCard, Model, ParamCard

    from pyamplicol.models.loading import _sanitized_model_environment

    asset_root = Path(str(_assets_resource()))
    for json_path in sorted((asset_root / "json").glob("*/restrict_*.json")):
        model_name = json_path.parent.name
        model_path = json_path.parent / f"{model_name}.json"
        ufo_path = asset_root / "ufo" / model_name / f"{json_path.stem}.dat"
        with _sanitized_model_environment():
            model = Model.from_json(model_path.read_text(encoding="utf-8"))
        parameter_names = {
            optionally_lower_external_parameter_name(parameter.name)
            for parameter in model.get_external_parameters()
        }
        expected = InputParamCard.from_param_card(
            ParamCard(str(ufo_path)),
            model=model,
        )
        actual = InputParamCard.from_json_file(str(json_path))

        assert set(actual) == parameter_names, json_path
        assert actual == InputParamCard(
            {name: expected[name] for name in parameter_names}
        ), json_path


def test_packaged_jsons_store_evaluated_unrestricted_defaults() -> None:
    from ufo_model_loader.commands import load_model

    for relative in EXPECTED_MODEL_SHAPES:
        path = Path(str(_assets_resource().joinpath("json", relative)))
        payload = json.loads(path.read_text(encoding="utf-8"))
        # These values are part of the shipped serialization, not supplied by
        # a pyAmpliCol loader workaround or filled only after reading the file.
        for entry in (*payload["parameters"], *payload["couplings"]):
            assert entry["value"] is not None, (relative, entry["name"])
            assert all(math.isfinite(value) for value in entry["value"])
        model, _card = load_model(str(path), "full", True)
        reloaded = json.loads(model.to_json())
        for field in ("parameters", "couplings"):
            recomputed = {entry["name"]: entry["value"] for entry in reloaded[field]}
            for entry in payload[field]:
                assert all(
                    math.isclose(a, b, rel_tol=2e-15, abs_tol=0.0)
                    for a, b in zip(
                        entry["value"], recomputed[entry["name"]], strict=True
                    )
                ), (relative, entry["name"])


def test_scalar_ufo_shape_is_environment_independent() -> None:
    scalars = _resource("ufo/scalars/parameters.py").read_text(encoding="utf-8")
    gravity = _resource("ufo/scalar_gravity/parameters.py").read_text(encoding="utf-8")
    combined = scalars + gravity
    assert "os.environ" not in combined
    assert "UFO_SCALARS_MODEL_" not in combined
    assert "UFO_GRAVITY_MODEL_" not in combined
    assert "N_SCALARS = 3" in scalars
    assert "N_POINT_INTERACTIONS = [3, 4, 5, 6, 7, 8, 9, 10]" in scalars
    assert "N_SCALARS = 3" in gravity
    assert "N_POINT_INTERACTIONS" not in gravity
