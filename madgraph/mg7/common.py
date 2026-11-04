"""Shared state of the mg7 launcher: the madspace import, logging and small helpers."""

import logging
import os
import re
import subprocess
import sys
import time

# Install madspace on first use, then import it. The bootstrap lives in its own
# module (which must not import madspace) so that an in-process caller can run
# it *before* importing this one -- see bootstrap.ensure_madspace.
from madgraph.mg7.bootstrap import (
    ensure_madspace as _ensure_madspace,
    drop_install_path as _drop_install_path,
    MADSPACE_DIR as _MADSPACE_DIR,
)

_ensure_madspace()

import madspace as ms

# madspace is imported; stop its install directory from shadowing the caller's
# yaml/packaging/... for the rest of what is now MG5's own session.
_drop_install_path()

_source_hash = subprocess.run(
    [sys.executable, str(_MADSPACE_DIR / "source_hash.py")],
    capture_output=True, text=True, check=True,
).stdout.strip()
if _source_hash != ms.SOURCE_HASH:
    print()
    print(
        "\033[1m\033[31mWARNING\033[39m: madspace source and installed binaries "
        "are not compatible (source hash mismatch) — consider recompiling "
        "madspace (e.g. `install madspace -y`)\033[0m"
    )
    print()

logger = logging.getLogger("madgraph7")
LOG_LEVEL_MAP = {
    ms.Logger.LogLevel.level_debug: logging.DEBUG,
    ms.Logger.LogLevel.level_info: logging.INFO,
    ms.Logger.LogLevel.level_warning: logging.WARNING,
    ms.Logger.LogLevel.level_error: logging.ERROR,
}
def ms_log_handler(level: ms.Logger.LogLevel, message: str):
    logger.log(LOG_LEVEL_MAP[level], message)
ms.Logger.set_log_handler(ms_log_handler)


def get_start_time():
    return time.time(), time.process_time()


def resolve_verbosity(verbosity: str) -> str:
    """Resolve the run_card "auto" verbosity to "pretty"/"log" depending on
    whether stdout is attached to a terminal; other values pass through
    unchanged."""
    if verbosity == "auto":
        return "pretty" if sys.stdout.isatty() else "log"
    return verbosity


def resolve_seed(seed: int) -> int:
    """Resolve the run_card "seed": -1 draws a fresh 64-bit seed via
    os.urandom, any other value is used as-is."""
    if seed == -1:
        return int.from_bytes(os.urandom(8), "big")
    return seed


def device_type_of(device_name: str) -> str:
    """The device type of one 'device' run_card entry (the optional ':<index>'
    suffix is stripped)."""
    return device_name.split(":")[0]


def backend_of(device_name: str, cpu_mode: str) -> str:
    """The build BACKEND that serves one 'device' run_card entry.

    ``cpu_mode`` describes the SIMD width of the CPU code and is therefore
    meaningless for the cuda/hip devices: those build the backend named after
    the device itself.
    """
    device_type = device_type_of(device_name)
    return cpu_mode if device_type == "cpu" else device_type


_PRECISION_2_FPTYPE = {"all64": "d", "all32": "f", "color32": "m", "denom64": "v"}


def fptype_of(precision: str) -> str:
    return _PRECISION_2_FPTYPE.get(str(precision).lower(), str(precision).lower())


def resolve_auto_backend(build_path: str) -> str:
    """Ask the matrix-element Makefile to resolve the ``auto`` cpu_mode.

    Given the produced shared library will have its name taken from the resolved
    backend name, we need to make sure the detection is taking place correctly
    and catch any possible error.
    """
    command = ["make", "-n", "BACKEND=auto", "detect-backend"]
    try:
        result = subprocess.run(
            command,
            cwd=build_path,
            capture_output=True,
            text=True,
            check=True,
        )
    except OSError as exc:
        raise RuntimeError(
            f"Could not run make to resolve cpu_mode='auto' in '{build_path}': {exc}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        output = "\n".join(
            part.strip() for part in (exc.stdout, exc.stderr) if part.strip()
        )
        detail = f"\nmake output:\n{output}" if output else ""
        raise RuntimeError(
            f"Could not resolve cpu_mode='auto' in '{build_path}': "
            f"Exit status {exc.returncode}.{detail}"
        ) from exc

    match = re.search(
        r"^BACKEND=(\S+) \(was auto\)$", result.stdout, re.MULTILINE
    )
    if match is None or match.group(1) == "auto":
        output = "\n".join(
            part.strip() for part in (result.stdout, result.stderr) if part.strip()
        )
        detail = f"\nmake output:\n{output}" if output else ""
        raise RuntimeError(
            f"Could not resolve cpu_mode='auto' in '{build_path}': "
            f"make failed to report a backend.{detail}"
        )
    return match.group(1)


def me_parameters(run_card):
    """Run-time parameters of the matrix-element libraries, passed to each
    instance through umami_set_parameter by ms.Context.load_matrix_element:
    the window of the $-excluded propagators is the run card bw_cutoff (the
    one the phase space uses around the resonances too), and with
    [generation] interference_helicity = "summed" the helicity sum as the
    |M|^2 of an interference (see interference_helicities_summed). That one is
    only sent when asked for: a library made before it existed refuses an
    unknown parameter."""
    parameters = {"bwcutoff": float(run_card["phasespace"]["bw_cutoff"])}
    if interference_helicities_summed(run_card):
        parameters["interference_helicity_summed"] = 1.
    return parameters


def mlm_pdf_ladder(run_card, mlm_clustering, is_decay=False, leptonic=False):
    """Whether an MLM sample's beam densities walk the clustering ladder, the
    pdf half of madevent's rewgt: under scale_scheme "madevent", and under
    "clustering_mean" with clustering_mean_pdf_ladder. The Python side of
    EnergyScale::mlm_pdf_reweighting, which decides the same for the kernel;
    the systematics then vary the ladder as madevent's do."""
    beam = run_card["beam"]
    return bool(
        mlm_clustering and not is_decay and not leptonic
        and beam["pdf_reweighting"]
        and (beam["scale_scheme"] == "madevent"
             or beam["clustering_mean_pdf_ladder"]))


def interference_helicities_summed(run_card):
    """Whether [generation] interference_helicity is "summed".

    For an interference (squared split orders dropping a component), a helicity
    can contribute negatively. "exact", the default, draws the helicity of each
    event on |T_i| and gives the event sign(T_i) * sum_j |T_j|, as madevent does:
    the helicity of the event is exact, at the cost of a larger sum|w| (about 2x
    the events for the same precision). "summed" keeps the helicity sum sum_j T_j
    as the weight (lower variance), and the helicity then means nothing: it is
    written as 9 in the LHE (lhe_helicities)."""
    return run_card["generation"].get("interference_helicity", "exact") == "summed"


def lhe_helicities(meta, run_card):
    """The helicity table the LHE completer writes for a subprocess: its own,
    or 9 for every particle of an interference subprocess whose |M|^2 is the
    helicity sum (interference_helicities_summed)."""
    helicities = meta["helicities"]
    if meta.get("interference") and interference_helicities_summed(run_card):
        return [[9] * len(row) for row in helicities]
    return helicities


def lhe_weight_info(status):
    """(IDWTUP, XMAXUP) for the LHE <init> block of a run with this generator
    status. The events are unweighted to +-sigma_abs, keeping the sign of the
    integrand, and XMAXUP is that unit weight rather than the signed cross
    section. A sample that can hold negative weights -- an interference --
    declares it with IDWTUP = -4, as madevent does: the cross section is then
    the mean of the signed event weights. With -3 a shower takes it as
    |XSECUP| times the mean sign instead (Pythia8 does), which is not the
    cross section, and +3 is for positive weights only. gridpack.py applies the
    same rule inline, since it must not import this module."""
    negative = status.mean < 0 or status.mean_abs > abs(status.mean) * (1 + 1e-12)
    return (-4 if negative else 3), status.mean_abs
