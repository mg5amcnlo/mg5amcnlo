# SPDX-License-Identifier: 0BSD
"""Prepare the developer-only scalar Higgs Effective Theory UFO model.

The upstream attachment has no package-manager identity and no explicit
redistribution terms were found.  This module therefore downloads and checks
the attachment in place for developer workflows; neither the archive nor the
prepared UFO belongs in a pyAmpliCol distribution.
"""

from __future__ import annotations

import ast
import hashlib
import io
import shutil
import tarfile
import tempfile
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEFT_UFO_URL = (
    "https://feynrules.irmp.ucl.ac.be/raw-attachment/wiki/"
    "HiggsEffectiveTheory/Higgs_Effective_Couplings_UFO.tgz"
)
HEFT_UFO_SHA256 = "c7492a8933c01482781936d5e07c64e4e72a2a78af026b6000084b64ec022356"
HEFT_UFO_ARCHIVE_ROOT = "Higgs_Effective_Couplings_UFO"
DEFAULT_HEFT_UFO_ROOT = ROOT / "dependencies" / "checkouts" / "heft-ufo"

_UPSTREAM_FILES = frozenset(
    {
        "Higgs_Effective_Couplings_UFO.log",
        "__init__.py",
        "couplings.py",
        "function_library.py",
        "lorentz.py",
        "object_library.py",
        "parameters.py",
        "particles.py",
        "vertices.py",
        "write_param_card.py",
    }
)
_PREPARED_FILES = _UPSTREAM_FILES | {"coupling_orders.py"}


class HEFTUFOPreparationError(RuntimeError):
    """The authenticated HEFT UFO could not be prepared safely."""


def _download(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            return response.read()
    except OSError as error:
        raise HEFTUFOPreparationError(f"cannot download HEFT UFO from {url}") from error


def _authenticated_archive(
    *,
    downloader: Callable[[str], bytes] = _download,
) -> bytes:
    payload = downloader(HEFT_UFO_URL)
    if not isinstance(payload, bytes):
        raise HEFTUFOPreparationError("HEFT UFO downloader returned non-byte content")
    actual = hashlib.sha256(payload).hexdigest()
    if actual != HEFT_UFO_SHA256:
        raise HEFTUFOPreparationError(
            "HEFT UFO archive SHA-256 mismatch: "
            f"expected {HEFT_UFO_SHA256}, got {actual}"
        )
    return payload


def _archive_files(archive: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    files: dict[str, tarfile.TarInfo] = {}
    root = HEFT_UFO_ARCHIVE_ROOT + "/"
    for member in archive.getmembers():
        name = member.name.rstrip("/")
        if name == HEFT_UFO_ARCHIVE_ROOT and member.isdir():
            continue
        if not member.name.startswith(root):
            raise HEFTUFOPreparationError(
                f"HEFT UFO archive member escapes its model root: {member.name!r}"
            )
        relative = member.name[len(root) :]
        if not relative or "/" in relative or member.issym() or member.islnk():
            raise HEFTUFOPreparationError(
                f"unexpected HEFT UFO archive member: {member.name!r}"
            )
        if not member.isfile() or relative in files:
            raise HEFTUFOPreparationError(
                f"unexpected HEFT UFO archive member: {member.name!r}"
            )
        files[relative] = member
    if set(files) != _UPSTREAM_FILES:
        missing = sorted(_UPSTREAM_FILES.difference(files))
        extra = sorted(set(files).difference(_UPSTREAM_FILES))
        raise HEFTUFOPreparationError(
            "unexpected HEFT UFO archive contents: "
            f"missing={missing!r}, extra={extra!r}"
        )
    return files


def _extract_authenticated_archive(payload: bytes, destination: Path) -> None:
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
            members = _archive_files(archive)
            for name, member in members.items():
                source = archive.extractfile(member)
                if source is None:
                    raise HEFTUFOPreparationError(
                        f"cannot read HEFT UFO archive member {member.name!r}"
                    )
                (destination / name).write_bytes(source.read())
    except (OSError, tarfile.TarError) as error:
        raise HEFTUFOPreparationError(
            "HEFT UFO attachment is not a valid tar.gz"
        ) from error


def _replace_exact(text: str, old: str, new: str, *, path: str) -> str:
    count = text.count(old)
    if count != 1:
        raise HEFTUFOPreparationError(
            f"expected one {old!r} occurrence in upstream {path}, found {count}"
        )
    return text.replace(old, new)


def _assignment_ranges(
    source: str, names: Iterable[str], *, path: str
) -> tuple[tuple[int, int], ...]:
    requested = set(names)
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as error:
        raise HEFTUFOPreparationError(
            f"upstream {path} is not valid Python 3"
        ) from error
    ranges: dict[str, tuple[int, int]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in requested:
            continue
        if node.end_lineno is None:
            raise HEFTUFOPreparationError(
                f"cannot locate assignment {target.id} in {path}"
            )
        ranges[target.id] = (node.lineno, node.end_lineno)
    if set(ranges) != requested:
        missing = sorted(requested.difference(ranges))
        raise HEFTUFOPreparationError(
            f"upstream {path} lacks expected assignments: {missing!r}"
        )
    return tuple(sorted(ranges.values()))


def _remove_assignments(source: str, names: Iterable[str], *, path: str) -> str:
    lines = source.splitlines(keepends=True)
    for start, end in reversed(_assignment_ranges(source, names, path=path)):
        while end < len(lines) and not lines[end].strip():
            end += 1
        del lines[start - 1 : end]
    return "".join(lines)


_COUPLING_ORDER_LIBRARY = """\

all_orders = []


class CouplingOrder(UFOBaseClass):
    require_args = ['name', 'expansion_order', 'hierarchy']

    def __init__(self, name, expansion_order, hierarchy, **options):
        UFOBaseClass.__init__(
            self, name, expansion_order, hierarchy, **options
        )
        all_orders.append(self)
"""

_COUPLING_ORDERS = """\
# Scalar HEFT coupling orders required by modern UFO consumers.
from object_library import all_orders, CouplingOrder


QCD = CouplingOrder(name='QCD', expansion_order=99, hierarchy=1)
QED = CouplingOrder(name='QED', expansion_order=99, hierarchy=2)
HIG = CouplingOrder(name='HIG', expansion_order=1, hierarchy=1)
HIW = CouplingOrder(name='HIW', expansion_order=1, hierarchy=1)
"""


def _prepare_sources(root: Path) -> None:
    object_library = (root / "object_library.py").read_text(encoding="utf-8")
    object_library = _replace_exact(
        object_library,
        "self.__dict__.iteritems()",
        "self.__dict__.items()",
        path="object_library.py",
    )
    if "all_orders = []" in object_library or "class CouplingOrder" in object_library:
        raise HEFTUFOPreparationError(
            "upstream object_library.py unexpectedly has coupling orders"
        )
    (root / "object_library.py").write_text(
        object_library.rstrip() + "\n" + _COUPLING_ORDER_LIBRARY,
        encoding="utf-8",
    )

    writer = (root / "write_param_card.py").read_text(encoding="utf-8")
    writer = _replace_exact(
        writer,
        "print 'write ./param_card.dat'",
        "print('write ./param_card.dat')",
        path="write_param_card.py",
    )
    (root / "write_param_card.py").write_text(writer, encoding="utf-8")

    parameters = (root / "parameters.py").read_text(encoding="utf-8")
    parameters = _replace_exact(
        parameters,
        "2.634632e7",
        "26346320",
        path="parameters.py",
    )
    parameters = _remove_assignments(
        parameters,
        ("MP", "WH1", "Gphi"),
        path="parameters.py",
    )
    (root / "parameters.py").write_text(parameters, encoding="utf-8")

    removals = {
        "particles.py": ("h1",),
        "vertices.py": ("V_19", "V_20"),
        "lorentz.py": ("VVS1", "VVVS1"),
        "couplings.py": ("GC_11", "GC_12"),
    }
    for name, assignments in removals.items():
        path = root / name
        path.write_text(
            _remove_assignments(
                path.read_text(encoding="utf-8"), assignments, path=name
            ),
            encoding="utf-8",
        )

    initializer = (root / "__init__.py").read_text(encoding="utf-8")
    initializer = _replace_exact(
        initializer,
        "import couplings\n",
        "import couplings\nimport coupling_orders\nimport function_library\n",
        path="__init__.py",
    )
    initializer = _replace_exact(
        initializer,
        "all_couplings = couplings.all_couplings\n",
        (
            "all_couplings = couplings.all_couplings\n"
            "all_orders = coupling_orders.all_orders\n"
        ),
        path="__init__.py",
    )
    (root / "__init__.py").write_text(initializer, encoding="utf-8")
    (root / "coupling_orders.py").write_text(_COUPLING_ORDERS, encoding="utf-8")


def validate_prepared_heft_ufo(root: Path) -> Path:
    """Validate the narrow scalar-only postconditions and return ``root``."""

    root = root.expanduser().resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise HEFTUFOPreparationError(f"prepared HEFT UFO is not a directory: {root}")
    files = {
        path.name for path in root.iterdir() if path.is_file() and not path.is_symlink()
    }
    if files != _PREPARED_FILES or any(
        path.is_dir() or path.is_symlink() for path in root.iterdir()
    ):
        raise HEFTUFOPreparationError("prepared HEFT UFO file inventory is invalid")

    for name in sorted(file for file in files if file.endswith(".py")):
        try:
            ast.parse((root / name).read_text(encoding="utf-8"), filename=name)
        except (OSError, SyntaxError) as error:
            raise HEFTUFOPreparationError(
                f"prepared HEFT UFO source is invalid: {name}"
            ) from error

    combined = "\n".join(
        (root / name).read_text(encoding="utf-8")
        for name in sorted(file for file in files if file.endswith(".py"))
    )
    forbidden = ("P.h1", "Gphi", "GC_11", "GC_12", "Epsilon(", "2.634632e7")
    observed = tuple(token for token in forbidden if token in combined)
    if observed:
        raise HEFTUFOPreparationError(
            f"prepared HEFT UFO retains excluded source content: {observed!r}"
        )
    if "26346320" not in combined:
        raise HEFTUFOPreparationError("prepared HEFT UFO lost the exact AH integer")

    particles = (root / "particles.py").read_text(encoding="utf-8")
    vertices = (root / "vertices.py").read_text(encoding="utf-8")
    lorentz = (root / "lorentz.py").read_text(encoding="utf-8")
    orders = (root / "coupling_orders.py").read_text(encoding="utf-8")
    for required in ("H = Particle", "G = Particle"):
        if required not in particles:
            raise HEFTUFOPreparationError(f"prepared HEFT UFO lacks {required}")
    for required in ("V_16 = Vertex", "V_17 = Vertex", "V_18 = Vertex"):
        if required not in vertices:
            raise HEFTUFOPreparationError(f"prepared HEFT UFO lacks {required}")
    for required in ("VVS3 = Lorentz", "VVVS2 = Lorentz", "VVVVS1 = Lorentz"):
        if required not in lorentz:
            raise HEFTUFOPreparationError(f"prepared HEFT UFO lacks {required}")
    for name in ("QCD", "QED", "HIG", "HIW"):
        if f"{name} = CouplingOrder" not in orders:
            raise HEFTUFOPreparationError(f"prepared HEFT UFO lacks {name} order")
    return root


def prepare_heft_ufo(
    destination: Path = DEFAULT_HEFT_UFO_ROOT,
    *,
    downloader: Callable[[str], bytes] = _download,
) -> Path:
    """Download, authenticate, scalarize, and install the HEFT UFO."""

    destination = destination.expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _authenticated_archive(downloader=downloader)
    with tempfile.TemporaryDirectory(
        prefix=".heft-ufo-prepare-", dir=destination.parent
    ) as temporary:
        prepared = Path(temporary) / "prepared"
        prepared.mkdir()
        _extract_authenticated_archive(payload, prepared)
        _prepare_sources(prepared)
        validate_prepared_heft_ufo(prepared)
        if destination.exists():
            if destination.is_symlink() or not destination.is_dir():
                raise HEFTUFOPreparationError(
                    "HEFT UFO destination is not a replaceable directory: "
                    f"{destination}"
                )
            shutil.rmtree(destination)
        prepared.replace(destination)
    return validate_prepared_heft_ufo(destination)


__all__ = [
    "DEFAULT_HEFT_UFO_ROOT",
    "HEFT_UFO_ARCHIVE_ROOT",
    "HEFT_UFO_SHA256",
    "HEFT_UFO_URL",
    "HEFTUFOPreparationError",
    "prepare_heft_ufo",
    "validate_prepared_heft_ufo",
]
