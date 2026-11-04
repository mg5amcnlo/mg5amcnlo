# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from tools.developer import heft_ufo


def _upstream_sources() -> dict[str, str]:
    return {
        "Higgs_Effective_Couplings_UFO.log": "upstream log\n",
        "__init__.py": ("import couplings\nall_couplings = couplings.all_couplings\n"),
        "couplings.py": (
            "GC_8 = Coupling(name='GC_8')\n"
            "GC_9 = Coupling(name='GC_9')\n"
            "GC_10 = Coupling(name='GC_10')\n"
            "GC_11 = Coupling(name='GC_11')\n"
            "GC_12 = Coupling(name='GC_12')\n"
        ),
        "function_library.py": "def complexconjugate(value):\n    return value\n",
        "lorentz.py": (
            "VVS3 = Lorentz(name='VVS3')\n"
            "VVVS2 = Lorentz(name='VVVS2')\n"
            "VVVVS1 = Lorentz(name='VVVVS1')\n"
            "VVVVS2 = Lorentz(name='VVVVS2')\n"
            "VVVVS3 = Lorentz(name='VVVVS3')\n"
            "VVS1 = Lorentz(name='VVS1', structure='Epsilon(1,2,3,4)')\n"
            "VVVS1 = Lorentz(name='VVVS1', structure='Epsilon(1,2,3,4)')\n"
        ),
        "object_library.py": (
            "class UFOBaseClass:\n"
            "    def values(self):\n"
            "        return self.__dict__.iteritems()\n"
        ),
        "parameters.py": (
            "AH = Parameter(name='AH', value=2.634632e7)\n"
            "MP = Parameter(name='MP', value=1)\n"
            "WH1 = Parameter(name='WH1', value=1)\n"
            "Gphi = Parameter(name='Gphi', value=1)\n"
        ),
        "particles.py": (
            "G = Particle(name='G')\nH = Particle(name='H')\nh1 = Particle(name='h1')\n"
        ),
        "vertices.py": (
            "V_16 = Vertex(name='V_16')\n"
            "V_17 = Vertex(name='V_17')\n"
            "V_18 = Vertex(name='V_18')\n"
            "V_19 = Vertex(name='V_19', particles=[P.h1])\n"
            "V_20 = Vertex(name='V_20', particles=[P.h1])\n"
        ),
        "write_param_card.py": "print 'write ./param_card.dat'\n",
    }


def _archive(sources: dict[str, str]) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for relative, source in sorted(sources.items()):
            payload = source.encode("utf-8")
            member = tarfile.TarInfo(f"{heft_ufo.HEFT_UFO_ARCHIVE_ROOT}/{relative}")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    return stream.getvalue()


def _accept_payload(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> None:
    monkeypatch.setattr(
        heft_ufo,
        "HEFT_UFO_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )


def test_prepare_heft_ufo_authenticates_and_scalarizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _archive(_upstream_sources())
    _accept_payload(monkeypatch, payload)

    destination = heft_ufo.prepare_heft_ufo(
        tmp_path / "heft-ufo",
        downloader=lambda _url: payload,
    )

    assert heft_ufo.validate_prepared_heft_ufo(destination) == destination
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in destination.glob("*.py")
    )
    assert "26346320" in combined
    assert "h1" not in combined
    assert "Epsilon(" not in combined
    assert "GC_11" not in combined
    assert "GC_12" not in combined
    assert "self.__dict__.items()" in combined
    assert "print('write ./param_card.dat')" in combined
    assert {"QCD", "QED", "HIG", "HIW"} <= {
        line.split()[0]
        for line in (destination / "coupling_orders.py")
        .read_text(encoding="utf-8")
        .splitlines()
        if " = CouplingOrder" in line
    }


def test_prepare_heft_ufo_rejects_wrong_archive_hash(
    tmp_path: Path,
) -> None:
    payload = _archive(_upstream_sources())
    with pytest.raises(heft_ufo.HEFTUFOPreparationError, match="SHA-256 mismatch"):
        heft_ufo.prepare_heft_ufo(
            tmp_path / "heft-ufo",
            downloader=lambda _url: payload,
        )


def test_prepare_heft_ufo_rejects_archive_inventory_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _upstream_sources()
    sources["unexpected.py"] = "pass\n"
    payload = _archive(sources)
    _accept_payload(monkeypatch, payload)

    with pytest.raises(
        heft_ufo.HEFTUFOPreparationError,
        match="unexpected HEFT UFO archive contents",
    ):
        heft_ufo.prepare_heft_ufo(
            tmp_path / "heft-ufo",
            downloader=lambda _url: payload,
        )


def test_prepare_heft_ufo_rejects_upstream_source_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _upstream_sources()
    sources["parameters.py"] = sources["parameters.py"].replace(
        "2.634632e7",
        "26346320.0",
    )
    payload = _archive(sources)
    _accept_payload(monkeypatch, payload)

    with pytest.raises(heft_ufo.HEFTUFOPreparationError, match="expected one"):
        heft_ufo.prepare_heft_ufo(
            tmp_path / "heft-ufo",
            downloader=lambda _url: payload,
        )
