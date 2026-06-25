#! /usr/bin/env python3

import madspace as ms
import os
import glob
import json
import tomllib
import argparse

def main() -> None:
    # load run card and metadata
    with open(os.path.join("Cards", "run_card.toml"), "rb") as f:
        run_card = tomllib.load(f)
    run_args = run_card["run"]
    gen_args = run_card["generation"]
    param_card_path = os.path.join("Cards", "param_card.dat")
    with open(os.path.join("data", "data.json")) as f:
        madspace_data = json.load(f)

    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=run_args["run_name"])
    parser.add_argument("--device", type=str, nargs="*")
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
        choices=["none", "pretty", "log"]
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

    # initialize event directory
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

    # initialize context
    device_names = args.device if args.device else run_args["devices"]
    contexts = []
    device_types = []
    for device_name in device_names:
        if ":" in device_name:
            device_type, device_index_str = device_name.split(":")
            device_index = int(device_index_str)
        else:
            device_type = device_name
            device_index = 0
        device_types.append(device_type)
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
    config.verbosity = args.verbosity
    config.combine_thread_count = run_args["combine_thread_pool_size"]
    config.cut_efficiency_threshold = gen_args["cut_efficiency_threshold"]
    config.max_cut_repetitions = gen_args["max_cut_repetitions"]

    # set up contexts
    global_dir = os.path.join("data", "globals")
    for context, device_type in zip(contexts, device_types):
        context.load_globals(global_dir)
        for me_path in madspace_data["matrix_elements"]:
            context.load_matrix_element(
                me_path.format(device=device_type), param_card_path
            )

    # set up generators
    channel_generators = [
        ms.ChannelEventGenerator.load(
            os.path.join("data", "channels", file),
            contexts,
            event_file=os.path.join(run_path, f"events.{name}.npy"),
            weight_file=os.path.join(run_path, f"weights.{name}.npy"),
            config=config,
        )
        for name, file in madspace_data["channels"].items()
    ]
    event_generator = ms.EventGenerator(
        contexts=contexts,
        channels=channel_generators,
        status_file=os.path.join(run_path, "info.json"),
        config=config,
    )

    # run generation
    event_generator.generate()
    output_format = args.output_format
    if output_format == "compact_npy":
        event_generator.combine_to_compact_npy(
            os.path.join(run_path, "events.npy")
        )
    elif output_format == "lhe_npy":
        lhe_completer = ms.LHECompleter.load(os.path.join("data", "lhe.json"))
        event_generator.combine_to_lhe_npy(
            os.path.join(run_path, "events.npy"), lhe_completer
        )
    elif output_format == "lhe":
        lhe_completer = ms.LHECompleter.load(os.path.join("data", "lhe.json"))
        event_generator.combine_to_lhe(
            os.path.join(run_path, "events.lhe"), lhe_completer
        )
    else:
        raise ValueError("Unknown output format")

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    try:
        main()
    except KeyboardInterrupt:
        pass
