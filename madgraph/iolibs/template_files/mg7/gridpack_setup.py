"""Gridpack setup shared by bin/generate_events and bin/event_stream.py: locating
madspace, reading the cards and data files, and building the contexts, channels,
LHE completer and systematics of the gridpack. Paths are resolved relative to
the gridpack directory, so nothing depends on the current directory."""

import json
import os
import subprocess
import sys
from pathlib import Path

GRIDPACK_DIR = Path(os.path.realpath(__file__)).parent.parent

# search order for madspace package:
#   1. a precompiled install (madspace/install/madspace)
#   2. bundled source that still needs to be built (madspace/install.py)
#   3. otherwise fall back to madspace is available in the environment
_LOCAL_MADSPACE_DIR = GRIDPACK_DIR / "madspace"
_LOCAL_INSTALL_DIR = _LOCAL_MADSPACE_DIR / "install"
if (_LOCAL_INSTALL_DIR / "madspace").is_dir():
    sys.path.insert(0, str(_LOCAL_INSTALL_DIR))
elif (_LOCAL_MADSPACE_DIR / "install.py").is_file():
    print()
    print("You don't have madspace installed for this gridpack")
    print("Running interactive madspace installation script")
    print()

    _result = subprocess.run([sys.executable, str(_LOCAL_MADSPACE_DIR / "install.py")])
    if _result.returncode != 0:
        raise RuntimeError("madspace installation failed — see output above")
    sys.path.insert(0, str(_LOCAL_INSTALL_DIR))

# MadtRex (event reweighting), if the gridpack was made with include_madtrex:
# put it on the path so it can be imported next to madspace. Nothing in this
# script imports it.
_LOCAL_MADTREX_INSTALL_DIR = GRIDPACK_DIR / "madtrex" / "install"
if (_LOCAL_MADTREX_INSTALL_DIR / "madtrex").is_dir():
    sys.path.insert(1, str(_LOCAL_MADTREX_INSTALL_DIR))

import madspace as ms

_PRECISION_2_FPTYPE = {"all64": "d", "all32": "f", "color32": "m", "denom64": "v"}


def fptype_of(precision: str) -> str:
    """[run] precision -> FPTYPE the matrix elements were compiled with (as
    fptype_of() in madgraph/mg7/common.py, which a gridpack can't import)."""
    return _PRECISION_2_FPTYPE.get(str(precision).lower(), str(precision).lower())


def gridpack_path(*parts) -> str:
    """A path inside the gridpack directory."""
    return str(GRIDPACK_DIR.joinpath(*parts))


def load_run_card():
    """The gridpack run card (Cards/grid_run_card.toml), as plain TOML: a
    gridpack must run without a MadGraph installation."""
    import tomllib
    with open(gridpack_path("Cards", "grid_run_card.toml"), "rb") as f:
        return tomllib.load(f)


def load_madspace_data() -> dict:
    """data/data.json: the matrix elements, their run-time parameters and the
    madspace source hash the gridpack was made with."""
    with open(gridpack_path("data", "data.json")) as f:
        return json.load(f)


def source_hash_matches(madspace_data: dict) -> bool:
    return madspace_data["source_hash"] == ms.SOURCE_HASH


SOURCE_HASH_MESSAGE = (
    "The madspace version is not identical to the one used to generate "
    "the gridpack. This can lead to errors or incorrect results"
)


def resolve_seed(seed: int) -> int:
    """Resolve the run_card "seed": -1 draws a fresh 64-bit seed via
    os.urandom, any other value is used as-is."""
    if seed == -1:
        return int.from_bytes(os.urandom(8), "big")
    return seed


def make_context(device_name: str, cpu_mode: str, cpu_thread_pool_size: int,
                 gpu_thread_pool_size: int):
    """The context for a device name like "cpu", "cuda:1" or "hip", with the
    backend its matrix elements are built for."""
    if ":" in device_name:
        device_type, device_index_str = device_name.split(":")
        device_index = int(device_index_str)
    else:
        device_type = device_name
        device_index = 0
    # cpu_mode names the SIMD width of the CPU code, so it applies to the
    # 'cpu' devices only: cuda/hip build the backend named after the device.
    backend = cpu_mode if device_type == "cpu" else device_type
    if device_type == "cuda":
        device = ms.cuda_device(device_index)
        pool_size = gpu_thread_pool_size
    elif device_type == "hip":
        device = ms.hip_device(device_index)
        pool_size = gpu_thread_pool_size
    else:
        device = ms.cpu_device()
        pool_size = cpu_thread_pool_size
    return ms.Context(device=device, thread_count=pool_size), backend


def load_context_data(contexts, backends, madspace_data: dict, precision: str) -> None:
    """Load the globals and the matrix elements of the gridpack into every
    context; precision is the run's, which selects the {fptype} libraries."""
    fptype = fptype_of(precision)
    for context in contexts:
        context.load_globals(gridpack_path("data", "globals.tar"))
    # custom functions come first, as in the run that made the gridpack
    from custom_function_loader import load_custom_function
    device_types = [b if b in ("cuda", "hip") else "cpu" for b in backends]
    for entry in madspace_data.get("custom_functions", []):
        # [kind, path, parameters]; older gridpacks store fewer entries
        entry = entry if isinstance(entry, list) else ["", entry]
        kind, path, parameters = (entry + [{}])[:3]
        load_custom_function(
            ms, contexts, device_types, kind, _gridpack_relative(path), parameters
        )
    # in the order of the run, which gave them their indices
    for context, backend in zip(contexts, backends):
        for path_format, init_path, parameters in madspace_data["matrix_elements"]:
            context.load_matrix_element(
                _gridpack_relative(path_format.format(device=backend, fptype=fptype)),
                _gridpack_relative(init_path),
                parameters,
            )


def load_channels(contexts, config, file_dir=None):
    """The channel generators stored in data/channels.json. With `file_dir`,
    their temporary event files are written there; without, events are only
    kept in memory."""
    with open(gridpack_path("data", "channels.json")) as f:
        channels = json.load(f)
    return [
        ms.ChannelEventGenerator.load_json(
            json.dumps(channel),
            contexts,
            event_file=os.path.join(file_dir, f"events.{name}.npy") if file_dir else "",
            weight_file=os.path.join(file_dir, f"weights.{name}.npy") if file_dir else "",
            config=config,
        )
        for name, channel in channels.items()
    ]


def load_weighting_groups(
    max_overweight_truncation, freeze_max_weight_after,
    first_pass_max_overweight_truncation, first_pass_freeze_max_weight_after,
):
    """The re- and unweighting steps after the first unweighting stored in
    data/groups.json, with the unweighting settings of the final pass. The
    first unweighting of a group with an unweighting step is a first pass."""
    with open(gridpack_path("data", "groups.json")) as f:
        groups = json.load(f)
    for group in groups:
        unweights = False
        for step in group["steps"]:
            step["max_overweight_truncation"] = max_overweight_truncation
            step["freeze_max_weight_after"] = freeze_max_weight_after
            unweights |= step["mode"] == "unweight"
        # the card replaces the stored values, which older gridpacks lack
        group.pop("first_pass_max_overweight_truncation", None)
        group.pop("first_pass_freeze_max_weight_after", None)
        if unweights:
            group["first_pass_max_overweight_truncation"] = (
                first_pass_max_overweight_truncation
            )
            group["first_pass_freeze_max_weight_after"] = first_pass_freeze_max_weight_after
    return ms.weighting_groups_from_json(json.dumps(groups))


def load_lhe_completer():
    return ms.LHECompleter.load(gridpack_path("data", "lhe.json"))


def build_lhe_meta(status, seed: int, systematics=None):
    """The LHE header/<init> metadata of this gridpack run: the cards, beams
    and PDF the gridpack was made with (data/lhe_meta.json), with the cross
    section (from the generator `status`) and seed of this run.
    ``systematics`` adds the <initrwgt> block, for the writers that do not
    inject it themselves (header.lhe); leave it unset for combine_to_lhe,
    which does."""
    headers = []
    data = {}
    meta_path = gridpack_path("data", "lhe_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            data = json.load(f)
        headers = [ms.LHEHeader(name=h["name"], content=h["content"],
                                escape_content=h["escape_content"])
                   for h in data.pop("headers")]
    headers.append(ms.LHEHeader(name="MG7Seed", content=str(seed)))
    if systematics is not None and systematics.weight_ids:
        headers.append(ms.LHEHeader(
            name="initrwgt", content=systematics.initrwgt(), escape_content=False
        ))
    # IDWTUP and XMAXUP of this run, as launch.lhe_weight_info: negative when
    # the sample can hold negative weights, and the unit weight sigma_abs
    negative = status.mean < 0 or status.mean_abs > abs(status.mean) * (1 + 1e-12)
    data["weight_mode"] = -4 if negative else 3
    return ms.LHEMeta(
        # positional: the pybind arg name for max_weight is non-kwarg-safe
        processes=[ms.LHEProcess(status.mean, status.error, status.mean_abs, 1)],
        headers=headers,
        **data,
    )


def _gridpack_relative(path: str) -> str:
    """Paths stored relative to the gridpack refer to the gridpack directory."""
    return path if os.path.isabs(path) else gridpack_path(path)


def _pdf_search_paths(stored_path):
    """Directories in which the LHAPDF grids may live: the LHAPDF_DATA_PATH
    entries, the paths known to the lhapdf module, then the directory the
    gridpack was created with."""
    paths = []
    env = os.environ.get("LHAPDF_DATA_PATH")
    if env:
        paths += env.split(os.pathsep)
    try:
        import lhapdf
        paths += list(lhapdf.paths())
    except Exception:
        pass
    if stored_path:
        paths.append(stored_path)
    return paths


def _locate_pdf_file(stored_file):
    """Find a PDF grid/info file: the stored path, else the same <set>/<file>
    relative to one of the PDF search paths."""
    stored_file = _gridpack_relative(stored_file)
    if os.path.exists(stored_file):
        return stored_file
    set_dir, name = os.path.split(stored_file)
    set_root, set_name = os.path.split(set_dir)
    for base in _pdf_search_paths(set_root):
        candidate = os.path.join(base, set_name, name)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "PDF file %s not found (set LHAPDF_DATA_PATH to the LHAPDF data directory)"
        % stored_file)


def load_systematics(run_card, backends=(), me_parameters=None):
    """Rebuild the ms.SystematicsCalculator saved with the gridpack
    (data/systematics.json) when [systematics] enable is set; None otherwise.
    The matrix elements of the mixed-order subprocesses are reloaded into a CPU
    context (when a CPU library is available) for the mu_R variations."""
    path = gridpack_path("data", "systematics.json")
    try:
        enabled = bool(run_card["systematics"]["enable"])
    except Exception:
        enabled = False
    if not enabled or not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    config = ms.SystematicsConfig.from_json(json.dumps(data["config"]))
    for spec in config.pdf_members:
        spec.grid_file = _locate_pdf_file(spec.grid_file)
        spec.info_file = _locate_pdf_file(spec.info_file)
    subproc_args = [
        ms.SubprocessSystArgs.from_json(json.dumps(a)) for a in data["subproc_args"]
    ]
    nominal_pdf = nominal_pdf2 = None
    if config.has_pdf:
        nominal_pdf = ms.PdfGrid(_locate_pdf_file(data["nominal_grid_file"]))
        # the second beam's set, recorded only when it differs from the first
        if data.get("nominal_grid_file2"):
            nominal_pdf2 = ms.PdfGrid(_locate_pdf_file(data["nominal_grid_file2"]))
    # The mu_R variations always need an alpha_s grid, so a gridpack that does
    # not name one cannot reweight anything -- drop the systematics rather than
    # die here (gridpacks written before the launcher recorded the file for a
    # run without parton luminosity).
    if not data.get("nominal_info_file"):
        print("WARNING systematics: the gridpack records no alpha_s .info file, "
              "the scale/PDF weights are dropped")
        return None
    nominal_alpha_s = ms.AlphaSGrid(_locate_pdf_file(data["nominal_info_file"]))
    # PDFs, alpha_s and matrix elements are evaluated on this CPU context
    context = ms.Context(device=ms.cpu_device(), thread_count=1)
    matrix_elements, flavor_remap = [], []
    need = [i for i, a in enumerate(subproc_args) if a.qcd_power < 0]
    # the CPU backend of this run (cpu_mode, resolved when the gridpack was
    # saved), else the one the launcher used
    backend = data.get("me_backend")
    if need and data.get("me_paths"):
        cpu = [b for b in backends if not str(b).startswith(("cuda", "hip")) and b != "auto"]
        if cpu:
            backend = cpu[0]
    if need and backend and data.get("me_paths"):
        flavor_remap = data.get("flavor_remap", [])
        for i, me_path in enumerate(data["me_paths"]):
            if i not in need:
                matrix_elements.append(None)
                continue
            lib = _gridpack_relative(me_path.format(
                device=backend, fptype=fptype_of(run_card["run"]["precision"])
            ))
            if not os.path.exists(lib):
                print("WARNING systematics: %s not found, mu_R variations of the "
                      "mixed-order subprocesses are dropped" % lib)
                matrix_elements, flavor_remap = [], []
                break
            api = context.load_matrix_element(
                lib, gridpack_path("Cards", "param_card.dat"), me_parameters or {}
            )
            matrix_elements.append(ms.MatrixElement(
                api,
                [ms.MatrixElement.momenta_in, ms.MatrixElement.alpha_s_in,
                 ms.MatrixElement.flavor_in],
                [ms.MatrixElement.matrix_element_out],
                False,
                # the frame the events were generated with, not whatever the
                # gridpack's run card says now: it has to match the matrix
                # element inside the saved channel generators
                data.get("me_frame", []),
                data.get("incoming_count", 2),
            ))
    systematics = ms.SystematicsCalculator(
        config, subproc_args, nominal_pdf, nominal_alpha_s,
        context, matrix_elements, flavor_remap, nominal_pdf2=nominal_pdf2)
    for warning in systematics.warnings:
        print("WARNING systematics: %s" % warning)
    return systematics
