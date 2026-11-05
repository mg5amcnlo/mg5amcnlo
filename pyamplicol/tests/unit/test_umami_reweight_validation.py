# SPDX-License-Identifier: 0BSD
"""Crafted UMAMI metadata must not reach the generated reweight sources."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pyamplicol.artifacts.umami_reweight import (
    load_umami_metadata,
    reweight_header,
    write_reweight_provider,
)


def _process(identifier="proc"):
    return {
        "id": identifier,
        "pdgs": [21, 21, 21, 21],
        "alpha_s_parameter": "aS",
        "helicity_ids": ["h0"],
    }


def _document(provider_id="p0", permutation=(0, 1, 2, 3), process_index=0):
    return {
        "kind": "pyamplicol-umami-metadata",
        "provider": {
            "id": provider_id,
            "particle_count": 4,
            "color_accuracy": "lc",
            "artifact_id": "a",
        },
        "runtime_processes": [_process()],
        "helicities": [[1, 1, 1, 1]],
        "channels": [
            {
                "processes": [
                    {
                        "runtime": {
                            "process_index": process_index,
                            "permutation": list(permutation),
                            "helicity_export": [0],
                        }
                    }
                ]
            }
        ],
    }


def _write(artifact: Path, document) -> Path:
    root = artifact / "API" / "umami"
    root.mkdir(parents=True)
    (root / "metadata.json").write_text(json.dumps(document), encoding="utf-8")
    return artifact


def test_valid_metadata_renders():
    header = reweight_header(_document(), [_document()], "lc", "full")
    assert "reweight_c0_f0_permutation[] = {0, 1, 2, 3};" in header


@pytest.mark.parametrize(
    "document",
    [
        _document(permutation=(0, 1, 2, "3}; void f(void) {}")),
        _document(permutation=(0, 1, 2, True)),
        _document(process_index="0"),
    ],
)
def test_non_integer_metadata_is_rejected(document):
    with pytest.raises(ValueError, match="integer"):
        reweight_header(document, [_document()], "lc", "full")


def test_provider_id_is_validated(tmp_path):
    lc = _write(tmp_path / "lc", _document(provider_id="$(shell touch PWNED)"))
    full = _write(tmp_path / "full", _document())
    with pytest.raises(ValueError, match="provider id"):
        write_reweight_provider(lc, full, tmp_path / "out")
    assert not (tmp_path / "out" / "providers.mk").exists()


def test_metadata_path_stays_in_artifact(tmp_path):
    (tmp_path / "secret.json").write_text("{}", encoding="utf-8")
    index = {
        "kind": "pyamplicol-umami-provider-index",
        "providers": [{"id": "p0", "metadata": "../../../secret.json"}],
    }
    artifact = _write(tmp_path / "lc", index)
    with pytest.raises(ValueError, match="leaves"):
        load_umami_metadata(artifact)
