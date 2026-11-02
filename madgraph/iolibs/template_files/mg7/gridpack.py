#! /usr/bin/env python3

import atexit
import gzip
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# search order for madspace package:
#   1. a precompiled install (madspace/install/madspace)
#   2. bundled source that still needs to be built (madspace/install.py)
#   3. otherwise fall back to madspace is available in the environment
_GRIDPACK_DIR = Path(os.path.realpath(__file__)).parent.parent
_LOCAL_MADSPACE_DIR = _GRIDPACK_DIR / "madspace"
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
_LOCAL_MADTREX_INSTALL_DIR = _GRIDPACK_DIR / "madtrex" / "install"
if (_LOCAL_MADTREX_INSTALL_DIR / "madtrex").is_dir():
    sys.path.insert(1, str(_LOCAL_MADTREX_INSTALL_DIR))

import madspace as ms
from custom_function_loader import load_custom_function
import glob
import json
import tomllib
import argparse

_PRECISION_2_FPTYPE = {"all64": "d", "all32": "f", "color32": "m", "denom64": "v"}


def fptype_of(precision: str) -> str:
    """[run] precision -> FPTYPE the matrix elements were compiled with (as
    fptype_of() in launch.py, which this standalone script can't import)."""
    return _PRECISION_2_FPTYPE.get(str(precision).lower(), str(precision).lower())


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


def compress_lhe(path: str, use_pigz="auto", cpu_threads: int = -1) -> None:
    """Compress the completed LHE with the same size/tool policy as misc.gzip.

    A gridpack must be runnable without a MadGraph installation. Its run card
    carries the compression preference, and positive CPU pool sizes also bound
    pigz's worker count.
    """
    if os.path.getsize(path) > 256e6:
        pigz = shutil.which("pigz") if use_pigz is not False else None
        tool = pigz or shutil.which("gzip")
        if tool:
            command = [tool]
            if pigz and cpu_threads > 0:
                command += ["-p", str(cpu_threads)]
            subprocess.run(command + ["-f", path], check=True)
            return

    with open(path, "rb") as fin, \
            gzip.open(path + ".gz", "wb", compresslevel=6) as fout:
        shutil.copyfileobj(fin, fout, 4 * 1024 * 1024)
    os.remove(path)


def build_lhe_meta(event_generator, seed: int, systematics=None):
    """The LHE header/<init> metadata of this gridpack run: the cards, beams
    and PDF the gridpack was made with (data/lhe_meta.json), with the cross
    section and seed of this run. ``systematics`` adds the <initrwgt> block,
    for the writers that do not inject it themselves (header.lhe); leave it
    unset for combine_to_lhe, which does."""
    headers = []
    data = {}
    meta_path = os.path.join("data", "lhe_meta.json")
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
    status = event_generator.status()
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


def write_lhe_header(run_path: str, meta) -> None:
    """Write header.lhe next to events.npy: the <header>/<init> blocks the
    npy formats otherwise drop, with no events."""
    writer = ms.LHEFileWriter(os.path.join(run_path, "header.lhe"), meta)
    del writer  # closes the file (writes the closing tag)


# directory the gridpack was called from; relative command line paths refer to it
_INVOCATION_DIR = os.getcwd()


def main() -> None:
    # load run card and metadata
    with open(os.path.join("Cards", "grid_run_card.toml"), "rb") as f:
        run_card = tomllib.load(f)
    run_args = run_card["run"]
    gen_args = run_card["generation"]
    grid_args = run_card.get("gridpack", {})
    param_card_path = os.path.join("Cards", "param_card.dat")
    with open(os.path.join("data", "data.json")) as f:
        madspace_data = json.load(f)

    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=run_args["run_name"])
    parser.add_argument(
        "--seed", type=int, default=run_args.get("seed", -1),
        help="every run is reproducible from its seed; -1 draws a fresh random "
             "seed each run instead of fixing one here (still recorded in the "
             "run's info.json)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="directory for the final output files instead of a new Events/<run> "
             "folder (default: the run card value; relative paths refer to the "
             "current directory)"
    )
    parser.add_argument(
        "--temp_output_dir", type=str, default=None,
        help="directory for the temporary npy files, 'auto' for a new temporary "
             "directory (default: the run card value, else the output directory)"
    )
    parser.add_argument(
        "--ignore_source_hash", action="store_true",
        help="run even if the madspace version differs from the one used to "
             "generate the gridpack (can lead to errors or incorrect results)"
    )
    parser.add_argument("--device", type=str, nargs="*")
    parser.add_argument(
        "--madspace_cpu_mode",
        type=str,
        default=run_args.get("madspace_cpu_mode", "auto"),
        choices=["auto", "scalar", "simd_128", "simd_256", "avx512y", "simd_512"],
        help="SIMD width of the phase-space sampling (madspace) on the 'cpu' devices",
    )
    parser.add_argument(
        "--cpu_thread_pool_size", type=int, default=run_args["cpu_thread_pool_size"]
    )
    parser.add_argument(
        "--gpu_thread_pool_size", type=int, default=run_args["gpu_thread_pool_size"]
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default=run_args["verbosity"],
        choices=["silent", "pretty", "log", "auto"]
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default=run_args["output_format"],
        choices=["lhe", "lhe_npy", "compact_npy"]
    )
    parser.add_argument("--events", type=int, default=gen_args["events"])
    parser.add_argument("--max_overweight_truncation", type=float, default=gen_args["max_overweight_truncation"])
    parser.add_argument("--freeze_max_weight_after", type=int, default=gen_args["freeze_max_weight_after"])
    parser.add_argument("--cpu_batch_size", type=int, default=gen_args["cpu_batch_size"])
    parser.add_argument("--gpu_batch_size", type=int, default=gen_args["gpu_batch_size"])
    args = parser.parse_args()

    if madspace_data["source_hash"] != ms.SOURCE_HASH:
        message = (
            "The madspace version is not identical to the one used to generate "
            "the gridpack. This can lead to errors or incorrect results"
        )
        if not args.ignore_source_hash:
            sys.exit(
                f"\033[1m\033[31mERROR\033[39m: {message}.\n"
                "Use --ignore_source_hash to run anyway.\033[0m"
            )
        print()
        print(f"\033[1m\033[31mWARNING\033[39m: {message}\033[0m")
        print()
    seed = resolve_seed(args.seed)
    # has to be set before the madspace CPU backend is first loaded
    ms.set_simd_mode(args.madspace_cpu_mode)

    # initialize output directories; command line paths are relative to the
    # invocation directory, run card paths to the gridpack
    def resolve_dir(cli_value, card_value):
        if (cli_value or card_value) == "auto":
            return "auto"
        if cli_value:
            return os.path.abspath(
                os.path.join(_INVOCATION_DIR, os.path.expanduser(cli_value)))
        return os.path.abspath(os.path.expanduser(card_value)) if card_value else None

    output_dir = resolve_dir(args.output_dir, grid_args.get("output_dir"))
    temp_dir = resolve_dir(args.temp_output_dir, grid_args.get("temp_output_dir"))
    if output_dir is not None:
        run_path = output_dir
        os.makedirs(run_path, exist_ok=True)
    else:
        run_name = args.run_name
        os.makedirs("Events", exist_ok=True)
        run_dir_prefix = os.path.join("Events", f"{run_name}_")
        existing_run_dirs = glob.glob(f"{run_dir_prefix}*")
        run_index = 1
        for run_dir in existing_run_dirs:
            run_index_str = run_dir[len(run_dir_prefix):]
            if run_index_str.isnumeric():
                run_index = max(run_index, int(run_index_str) + 1)
        while True:
            try:
                run_path = f"{run_dir_prefix}{run_index:02d}"
                os.mkdir(run_path)
                break
            except FileExistsError:
                run_index += 1
    if temp_dir == "auto":
        temp_dir = tempfile.mkdtemp(prefix="mg7_events_")
        atexit.register(shutil.rmtree, temp_dir, ignore_errors=True)
    elif temp_dir is None:
        temp_dir = run_path
    else:
        os.makedirs(temp_dir, exist_ok=True)

    # initialize context
    device_names = args.device if args.device else run_args["device"]
    cpu_mode = run_args["cpu_mode"]
    contexts = []
    backends = []
    for device_name in device_names:
        if ":" in device_name:
            device_type, device_index_str = device_name.split(":")
            device_index = int(device_index_str)
        else:
            device_type = device_name
            device_index = 0
        # cpu_mode names the SIMD width of the CPU code, so it applies to the
        # 'cpu' devices only: cuda/hip build the backend named after the device.
        backends.append(cpu_mode if device_type == "cpu" else device_type)
        if device_type == "cuda":
            device = ms.cuda_device(device_index)
            pool_size = args.gpu_thread_pool_size
        elif device_type == "hip":
            device = ms.hip_device(device_index)
            pool_size = args.gpu_thread_pool_size
        else:
            device = ms.cpu_device()
            pool_size = args.cpu_thread_pool_size
        contexts.append(ms.Context(device=device, thread_count=pool_size))

    # set up generator configuration
    config = ms.GeneratorConfig()
    config.target_count = args.events
    config.max_overweight_truncation = args.max_overweight_truncation
    config.freeze_max_weight_after = args.freeze_max_weight_after
    config.cpu_batch_size = args.cpu_batch_size
    config.gpu_batch_size = args.gpu_batch_size
    config.verbosity = resolve_verbosity(args.verbosity)
    config.combine_thread_count = run_args["combine_thread_pool_size"]
    config.cut_efficiency_threshold = gen_args["cut_efficiency_threshold"]
    config.max_cut_repetitions = gen_args["max_cut_repetitions"]

    # set up contexts
    # run-time matrix-element parameters (bwcutoff) of the run that made the
    # gridpack; gridpacks written before they were recorded used the default
    me_parameters = madspace_data.get("me_parameters", {})
    # the FPTYPE the matrix elements were compiled with ({fptype} in me_path)
    fptype = fptype_of(run_args["precision"])
    globals_path = os.path.join("data", "globals.tar")
    for context in contexts:
        context.load_globals(globals_path)
    # custom functions come first, as in the run that made the gridpack
    device_types = [b if b in ("cuda", "hip") else "cpu" for b in backends]
    for entry in madspace_data.get("custom_functions", []):
        # [kind, path, parameters]; older gridpacks store fewer entries
        entry = entry if isinstance(entry, list) else ["", entry]
        kind, path, parameters = (entry + [{}])[:3]
        load_custom_function(ms, contexts, device_types, kind, path, parameters)
    for context, backend in zip(contexts, backends):
        for me_path in madspace_data["matrix_elements"]:
            context.load_matrix_element(
                me_path.format(device=backend, fptype=fptype), param_card_path, me_parameters
            )

    # set up generators
    with open(os.path.join("data", "channels.json")) as f:
        channels = json.load(f)
    channel_generators = [
        ms.ChannelEventGenerator.load_json(
            json.dumps(channel),
            contexts,
            event_file=os.path.join(temp_dir, f"events.{name}.npy"),
            weight_file=os.path.join(temp_dir, f"weights.{name}.npy"),
            config=config,
        )
        for name, channel in channels.items()
    ]
    event_generator = ms.EventGenerator(
        contexts=contexts,
        channels=channel_generators,
        status_file=ms.StatusFile(os.path.join(run_path, "info.json")),
        config=config,
        seed=seed,
    )

    # scale/PDF systematics (as configured when the gridpack was made)
    systematics = load_systematics(run_card, backends, param_card_path,
                                   me_parameters)

    # run generation
    event_generator.generate()
    output_format = args.output_format
    if output_format == "compact_npy":
        event_generator.combine_to_compact_npy(
            os.path.join(run_path, "events.npy"), systematics
        )
        write_lhe_header(run_path,
                         build_lhe_meta(event_generator, seed, systematics))
        # what npy_to_lhe needs to complete these events into LHE later
        shutil.copy(os.path.join("data", "lhe.json"),
                    os.path.join(run_path, "lhe_completer.json"))
    elif output_format == "lhe_npy":
        lhe_completer = ms.LHECompleter.load(os.path.join("data", "lhe.json"))
        event_generator.combine_to_lhe_npy(
            os.path.join(run_path, "events.npy"), lhe_completer, systematics
        )
        write_lhe_header(run_path,
                         build_lhe_meta(event_generator, seed, systematics))
    elif output_format == "lhe":
        lhe_completer = ms.LHECompleter.load(os.path.join("data", "lhe.json"))
        lhe_path = os.path.join(run_path, "events.lhe")
        # the cross section is known once generate() has converged, which is
        # before combine_to_lhe writes the <init> block
        event_generator.combine_to_lhe(
            lhe_path, lhe_completer, build_lhe_meta(event_generator, seed),
            systematics
        )
        compress_lhe(lhe_path, run_args.get("use_pigz", "auto"),
                     args.cpu_thread_pool_size)
    else:
        raise ValueError("Unknown output format")
    if systematics is not None:
        data = json.loads(systematics.summary())
        data["initrwgt"] = systematics.initrwgt()
        data["columns"] = ["rwgt_%d" % i for i in systematics.weight_ids]
        with open(os.path.join(run_path, "events.weights.json"), "w") as f:
            json.dump(data, f, indent=1)


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
    """Find a PDF grid/info file: the stored absolute path, else the same
    <set>/<file> relative to one of the PDF search paths."""
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


def load_systematics(run_card, backends=(), param_card_path=None,
                     me_parameters=None):
    """Rebuild the ms.SystematicsCalculator saved with the gridpack
    (data/systematics.json) when [systematics] enable is set; None otherwise.
    The matrix elements of the mixed-order subprocesses are reloaded into a CPU
    context (when a CPU library is available) for the mu_R variations."""
    path = os.path.join("data", "systematics.json")
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
            lib = me_path.format(
                device=backend, fptype=fptype_of(run_card["run"]["precision"])
            )
            if not os.path.exists(lib):
                print("WARNING systematics: %s not found, mu_R variations of the "
                      "mixed-order subprocesses are dropped" % lib)
                matrix_elements, flavor_remap = [], []
                break
            api = context.load_matrix_element(lib, param_card_path,
                                              me_parameters or {})
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

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    try:
        main()
    except KeyboardInterrupt:
        pass
