#! /usr/bin/env python3

import argparse
import atexit
import glob
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile

# bin/gridpack_setup.py, next to this script: also locates madspace
from gridpack_setup import (
    SOURCE_HASH_MESSAGE,
    build_lhe_meta,
    configure_mlm_shower_start,
    load_channels,
    load_context_data,
    load_lhe_completer,
    load_madspace_data,
    load_run_card,
    load_systematics,
    load_weighting_groups,
    make_context,
    resolve_seed,
    source_hash_matches,
)
import madspace as ms


def resolve_verbosity(verbosity: str) -> str:
    """Resolve the run_card "auto" verbosity to "pretty"/"log" depending on
    whether stdout is attached to a terminal; other values pass through
    unchanged."""
    if verbosity == "auto":
        return "pretty" if sys.stdout.isatty() else "log"
    return verbosity


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


def write_lhe_header(run_path: str, meta) -> None:
    """Write header.lhe next to events.npy: the <header>/<init> blocks the
    npy formats otherwise drop, with no events."""
    writer = ms.LHEFileWriter(os.path.join(run_path, "header.lhe"), meta)
    del writer  # closes the file (writes the closing tag)


# directory the gridpack was called from; relative command line paths refer to it
_INVOCATION_DIR = os.getcwd()


def main() -> None:
    run_card = load_run_card()
    run_args = run_card["run"]
    gen_args = run_card["generation"]
    grid_args = run_card.get("gridpack", {})
    amplicol_args = run_card.get("amplicol", {})
    madspace_data = load_madspace_data()

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
    # these two set the final pass, which gives the events
    parser.add_argument("--max_overweight_truncation", type=float, default=gen_args["max_overweight_truncation"])
    parser.add_argument("--freeze_max_weight_after", type=int, default=gen_args["freeze_max_weight_after"])
    parser.add_argument(
        "--first_pass_max_overweight_truncation", type=float,
        default=amplicol_args.get("first_pass_max_overweight_truncation", 0.01),
        help="max_overweight_truncation of the leading-colour unweighting of "
             "[amplicol] processes unweighted to full colour"
    )
    parser.add_argument(
        "--first_pass_freeze_max_weight_after", type=int,
        default=amplicol_args.get("first_pass_freeze_max_weight_after", 10000),
        help="freeze_max_weight_after of the leading-colour unweighting of "
             "[amplicol] processes unweighted to full colour"
    )
    parser.add_argument("--cpu_batch_size", type=int, default=gen_args["cpu_batch_size"])
    parser.add_argument("--gpu_batch_size", type=int, default=gen_args["gpu_batch_size"])
    args = parser.parse_args()

    if not source_hash_matches(madspace_data):
        message = SOURCE_HASH_MESSAGE
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
    contexts = []
    backends = []
    for device_name in device_names:
        context, backend = make_context(
            device_name,
            run_args["cpu_mode"],
            args.cpu_thread_pool_size,
            args.gpu_thread_pool_size,
        )
        contexts.append(context)
        backends.append(backend)

    # set up generator configuration
    config = ms.GeneratorConfig()
    config.target_count = args.events
    groups = load_weighting_groups(
        args.max_overweight_truncation, args.freeze_max_weight_after,
        args.first_pass_max_overweight_truncation, args.first_pass_freeze_max_weight_after,
    )
    config.max_overweight_truncation = args.max_overweight_truncation
    config.freeze_max_weight_after = args.freeze_max_weight_after
    config.cpu_batch_size = args.cpu_batch_size
    config.gpu_batch_size = args.gpu_batch_size
    config.verbosity = resolve_verbosity(args.verbosity)
    config.combine_thread_count = run_args["combine_thread_pool_size"]
    config.cut_efficiency_threshold = gen_args["cut_efficiency_threshold"]
    config.max_cut_repetitions = gen_args["max_cut_repetitions"]
    configure_mlm_shower_start(config, madspace_data)

    # set up contexts and generators
    load_context_data(contexts, backends, madspace_data, run_args["precision"])
    channel_generators = load_channels(contexts, config, temp_dir)
    event_generator = ms.EventGenerator(
        contexts=contexts,
        channels=channel_generators,
        status_file=ms.StatusFile(os.path.join(run_path, "info.json")),
        config=config,
        seed=seed,
        groups=groups,
    )

    # scale/PDF systematics (as configured when the gridpack was made)
    systematics = load_systematics(run_card, backends,
                                   madspace_data.get("me_parameters", {}))

    # run generation
    event_generator.generate()
    output_format = args.output_format
    if output_format == "compact_npy":
        event_generator.combine_to_compact_npy(
            os.path.join(run_path, "events.npy"), systematics
        )
        write_lhe_header(run_path,
                         build_lhe_meta(event_generator.status(), seed, systematics))
        # what npy_to_lhe needs to complete these events into LHE later
        shutil.copy(os.path.join("data", "lhe.json"),
                    os.path.join(run_path, "lhe_completer.json"))
    elif output_format == "lhe_npy":
        lhe_completer = load_lhe_completer()
        event_generator.combine_to_lhe_npy(
            os.path.join(run_path, "events.npy"), lhe_completer, systematics
        )
        write_lhe_header(run_path,
                         build_lhe_meta(event_generator.status(), seed, systematics))
    elif output_format == "lhe":
        lhe_completer = load_lhe_completer()
        lhe_path = os.path.join(run_path, "events.lhe")
        # the cross section is known once generate() has converged, which is
        # before combine_to_lhe writes the <init> block
        event_generator.combine_to_lhe(
            lhe_path, lhe_completer, build_lhe_meta(event_generator.status(), seed),
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


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    try:
        main()
    except KeyboardInterrupt:
        pass
