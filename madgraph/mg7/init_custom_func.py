"""Create a skeleton for a custom cut, scale, PDF or bias function.

The function is written for a single phase-space point in C, C++, Fortran,
CUDA or HIP. A generated UMAMI wrapper calls it on batches of events, so that
madspace can load it like a matrix element. The files go into
``custom_<function>/`` in the process directory, and ``make`` there builds
``lib/custom_<function>.so``. The run card option
``[custom_functions] <function>`` takes the path of that library.

Usage::

    bin/init_custom_func [-f {cut,scale,pdf,bias}] [-l {c,cpp,fortran,cuda,hip}]
                         [-y] [--force] [--build | --no-build]

Options that are not given are asked interactively, unless ``-y`` is passed.
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys

logger = logging.getLogger("madevent")

_TEMPLATE_FILES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "iolibs", "template_files"
)
TEMPLATE_DIR = os.path.join(_TEMPLATE_FILES, "mg7", "custom_functions")
UMAMI_HEADER = os.path.join(_TEMPLATE_FILES, "madmatrix", "umami.h")

FUNCTIONS = {
    "cut": (["UMAMI_OUT_CUT"], "a cut, true keeps the event and false rejects it"),
    "scale": (
        ["UMAMI_OUT_REN_SCALE", "UMAMI_OUT_FACT_SCALE_1", "UMAMI_OUT_FACT_SCALE_2"],
        "the renormalization and factorization scales",
    ),
    "pdf": (["UMAMI_OUT_PDF"], "a parton density x f(x, Q)"),
    "bias": (["UMAMI_OUT_BIAS"], "an event bias for the unweighting"),
}
# example parameter of parameters.h: (name, default, run card value)
EXAMPLE_PARAMETERS = {
    "cut": ("pt_min", "20.", "30.0"),
    "scale": ("ht_fraction", "0.5", "0.25"),
    "pdf": ("gluon_norm", "3.", "2.5"),
    "bias": ("pt0", "100.", "200.0"),
}
# language: (body template extension, description)
LANGUAGES = {
    "c": ("c", "C"),
    "cpp": ("cpp", "C++"),
    "fortran": ("F90", "Fortran"),
    "cuda": ("cu", "CUDA, for runs on NVIDIA GPUs"),
    "hip": ("cu", "HIP, for runs on AMD GPUs"),
    "python-numpy": ("py", "Python with numpy, for CPU runs, no compilation"),
    "python-torch": ("py", "Python with PyTorch, for CPU or GPU runs, no compilation"),
}

_NONINTERACTIVE = (not sys.stdin.isatty()) or any(a in ("-y", "--yes") for a in sys.argv[1:])


def _ask(question, default, choices):
    """Ask through MG5's cmd.ask when available, else input()."""
    if _NONINTERACTIVE:
        return default
    try:
        from madgraph.interface.extended_cmd import Cmd

        answer = Cmd().ask(question, default, choices=list(choices), timeout=0)
        return default if answer is None else str(answer)
    except Exception:
        pass
    try:
        answer = input(f"{question} [{default}]: ").strip()
    except EOFError:
        return default
    return answer or default


def ask_choice(title, options, default):
    """Numbered menu over ``options`` (name -> description), returns a name."""
    names = list(options)
    lines = [title]
    width = max(len(name) for name in names)
    for i, name in enumerate(names, 1):
        lines.append(f"  {i}. {name:{width}s} {options[name]}")
    print("\n".join(lines))
    choices = names + [str(i) for i in range(1, len(names) + 1)]
    while True:
        answer = _ask("Enter a number or a name", default, choices).strip().lower()
        if answer in names:
            return answer
        if answer.isdigit() and 1 <= int(answer) <= len(names):
            return names[int(answer) - 1]
        print(f"  Please enter one of {', '.join(names)} or 1-{len(names)}.")


def ask_yes_no(question, default=True):
    default_s = "y" if default else "n"
    while True:
        answer = _ask(question, default_s, ["y", "n"]).strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("  Please enter 'y' or 'n'.")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="bin/init_custom_func",
        description=(
            "Create a skeleton for a custom cut, scale, PDF or bias function in "
            "custom_<function>/. Compiled languages are built into "
            "lib/custom_<function>.so, Python functions need no build."
        ),
    )
    parser.add_argument("-f", "--function", choices=list(FUNCTIONS))
    parser.add_argument("-l", "--language", choices=list(LANGUAGES))
    parser.add_argument(
        "-y", "--yes", action="store_true",
        help="do not ask, use the defaults (cut, cpp) for options that are not given",
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing custom_<function> directory"
    )
    build = parser.add_mutually_exclusive_group()
    build.add_argument("--build", dest="build", action="store_true", default=None,
                       help="run make after creating the files (default)")
    build.add_argument("--no-build", dest="build", action="store_false")
    run_card = parser.add_mutually_exclusive_group()
    run_card.add_argument("--set-run-card", dest="set_run_card", action="store_true",
                          default=None, help="enable the function in Cards/run_card.toml (default)")
    run_card.add_argument("--no-set-run-card", dest="set_run_card", action="store_false")
    return parser


def is_python(language):
    return language.startswith("python-")


def function_path(function, language):
    """The run card path of the function, relative to the process directory."""
    if is_python(language):
        return os.path.join(f"custom_{function}", f"custom_{function}.py")
    return os.path.join("lib", f"custom_{function}.so")


def write_files(out_dir, function, language):
    """Write the function body and, for compiled languages, the wrapper,
    umami.h and the Makefile."""
    extension = LANGUAGES[language][0]
    body_file = f"custom_{function}.{extension}"
    if is_python(language):
        framework = language.split("-", 1)[1]
        os.makedirs(out_dir, exist_ok=True)
        shutil.copy(
            os.path.join(TEMPLATE_DIR, f"{function}_{framework}.py"),
            os.path.join(out_dir, body_file),
        )
        return body_file
    values = {
        "kind": function,
        "function_name": f"custom_{function}",
        "output_keys": ", ".join(FUNCTIONS[function][0]),
        "is_pdf": int(function == "pdf"),
        "is_scale": int(function == "scale"),
        "return_type": "bool" if function == "cut" else "double",
        "body_file": body_file,
        "language": language,
        "example": EXAMPLE_PARAMETERS[function][0],
        "example_default": EXAMPLE_PARAMETERS[function][1],
        "example_value": EXAMPLE_PARAMETERS[function][2],
    }
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(os.path.join(TEMPLATE_DIR, f"{function}.{extension}"), os.path.join(out_dir, body_file))
    shutil.copy(UMAMI_HEADER, os.path.join(out_dir, "umami.h"))
    wrapper = "wrapper_gpu.cu" if language in ("cuda", "hip") else "wrapper_cpu.c"
    for template in (wrapper, "Makefile", "parameters.h"):
        with open(os.path.join(TEMPLATE_DIR, template)) as f:
            text = f.read() % values
        with open(os.path.join(out_dir, template), "w") as f:
            f.write(text)
    return body_file


def build(out_dir):
    """Run make in ``out_dir``, logging to compile.log. Returns True on success."""
    log_path = os.path.join(out_dir, "compile.log")
    with open(log_path, "w") as log:
        result = subprocess.run(["make"], cwd=out_dir, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        with open(log_path) as log:
            print(log.read())
        print(f"Building failed, see {log_path}")
        return False
    return True


def set_run_card(me_dir, function, lib_path):
    from madgraph.various.banner import RunCardMG7

    card_path = os.path.join(me_dir, "Cards", "run_card.toml")
    run_card = RunCardMG7(card_path)
    run_card.set(f"custom_functions.{function}", lib_path)
    run_card.write(card_path)


def main(me_dir=None, argv=None):
    global _NONINTERACTIVE
    args = build_parser().parse_args(argv)
    if args.yes:
        _NONINTERACTIVE = True
    me_dir = me_dir or os.getcwd()

    function = args.function or ask_choice(
        "Which custom function do you want to create?",
        {name: desc for name, (_, desc) in FUNCTIONS.items()},
        "cut",
    )
    language = args.language or ask_choice(
        "Which language do you want to write it in?",
        {name: desc for name, (_, desc) in LANGUAGES.items()},
        "cpp",
    )

    out_dir = os.path.join(me_dir, f"custom_{function}")
    if os.path.exists(out_dir):
        if not (args.force or (not _NONINTERACTIVE and ask_yes_no(
                f"{out_dir} already exists. Overwrite it?", False))):
            print(f"{out_dir} already exists, use --force to overwrite it.")
            return 1
        shutil.rmtree(out_dir)

    body_file = write_files(out_dir, function, language)
    lib_path = function_path(function, language)
    print(f"Created {os.path.relpath(out_dir, me_dir)}/. Write your function in "
          f"{os.path.relpath(os.path.join(out_dir, body_file), me_dir)}.")

    do_build = args.build
    if is_python(language):
        if do_build:
            print("Python functions need no build.")
        do_build = False
    elif do_build is None:
        do_build = ask_yes_no("Build the default function now?", True)
    if do_build and not build(out_dir):
        return 1

    do_set = args.set_run_card
    if do_set is None:
        do_set = ask_yes_no(f"Set custom_functions.{function} = \"{lib_path}\" in the run card?", True)
    if do_set:
        set_run_card(me_dir, function, lib_path)
        print(f"Enabled it in Cards/run_card.toml: custom_functions.{function} = \"{lib_path}\".")
    else:
        print(f"To enable it, set custom_functions.{function} = \"{lib_path}\" in Cards/run_card.toml.")
    if not do_build and not is_python(language):
        print(f"Build it with `make -C {os.path.relpath(out_dir, me_dir)}`. "
              "bin/generate_events also rebuilds it before a run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
