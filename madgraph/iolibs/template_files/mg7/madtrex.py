"""MadtRex reweighting from within MadGraph.

The reweighting itself lives in madtrex_driver.py, which runs without
MadGraph (and is what gridpacks ship as bin/reweight). This module is the
MadGraph side of it: it adds what does need MadGraph -- regenerating the
process in another model for a "change model" reweight card, via
bin/madgraph -- and is what launch.py's --reweight uses.

Run with: python -m madgraph.iolibs.template_files.mg7.madtrex --help
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import madgraph as _mg_pkg

from . import param_reweight_card as param_rwgt_card
from . import madtrex_driver as _driver
from .madtrex_driver import (  # noqa: F401 -- re-exported
    ms,
    mtx,
    find_latest_run_dir,
    find_run_events,
    prepare_events,
    read_init_node,
    resolve_pdf_set_files,
    default_output_path,
    run_pdf_reweighting,
)

_MG_ROOT = Path(_mg_pkg.__file__).parents[1]

logger = logging.getLogger("madgraph7")


def regenerate_process_in_model(
    process_directory: str,
    model_spec: str,
    process_definition: str,
    rw_subdir: str = "rw_me",
    fptype: str = param_rwgt_card.DEFAULT_FPTYPE,
) -> str:
    """Regenerates process_definition in model_spec (a UFO model name, plus
    any further "import model" arguments, eg a restriction card) into
    process_directory/rw_subdir via a headless bin/madgraph run -- same
    subprocess-with-a-command-file pattern as compute_auto_widths() in
    launch.py -- then compiles every resulting subprocess, at fptype, for the
    devices process_directory's own run_card.toml requests (see
    madtrex_driver._compile_subprocesses()).

    Skips both steps if rw_subdir was already regenerated for this exact
    (model_spec, process_definition) pair (tracked via a marker file), so
    repeated reweighting runs against the same alternate model don't pay the
    regenerate cost every time -- note this means fptype only actually
    drives a compile on whichever call first regenerates a given rw_subdir;
    a later call requesting a different fptype against an already-cached
    rw_subdir gets the directory back unchanged, uncompiled for that fptype
    (run_parameter_reweighting() accounts for this itself by always calling
    _compile_subprocesses() again for the fptype it actually needs,
    regardless of whether this function regenerated anything). Returns
    rw_subdir's full path, ready to pass to Driver.load_process_directory()."""
    rw_dir = os.path.join(process_directory, rw_subdir)
    marker_path = os.path.join(rw_dir, ".madtrex_source.json")
    # The libraries are built with a LIBTAG of their own (see madmatrix.mk):
    # they are loaded next to the original process's, for the
    # normalisation, and the dynamic loader would take one of the same name
    # (the same model's common library, say) for the original's.
    libtag = "_madtrex_" + _driver._sanitize_dirname(rw_subdir)
    marker = {"model": model_spec, "process": process_definition, "libtag": libtag}
    if os.path.isfile(marker_path):
        with open(marker_path) as f:
            if json.load(f) == marker:
                return rw_dir
        shutil.rmtree(rw_dir)

    mg5 = str(_MG_ROOT / "bin" / "madgraph")
    if not os.path.exists(mg5):
        raise RuntimeError(
            f"cannot find bin/madgraph at {mg5}; needed to regenerate "
            f"'{process_definition}' in model '{model_spec}'"
        )
    cmds = (
        f"import model {model_spec}\n"
        f"generate {process_definition}\n"
        f"output mg7 {os.path.abspath(rw_dir)} -f\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".mg5", delete=False) as fh:
        fh.write(cmds)
        cmdfile = fh.name
    try:
        logger.info(
            "MadtRex: regenerating '%s' in model '%s' -> %s",
            process_definition, model_spec, rw_dir,
        )
        proc = subprocess.run([mg5, cmdfile], capture_output=False, text=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if proc.returncode != 0:
            raise RuntimeError(
                f"regenerating '{process_definition}' in model '{model_spec}' failed "
                f"(bin/madgraph exited with {proc.returncode})"
            )
    finally:
        os.remove(cmdfile)

    # the libraries' names with the LIBTAG, for whoever loads them
    json_path = os.path.join(rw_dir, "SubProcesses", "subprocesses.json")
    with open(json_path) as f:
        subprocesses = json.load(f)
    for entry in subprocesses:
        directory, name = os.path.split(entry["me_path"])
        if not name.startswith("libmadmatrix_"):
            raise RuntimeError(f"unexpected matrix element library name {entry['me_path']!r}")
        entry["me_path"] = os.path.join(directory, "libmadmatrix" + libtag + name[len("libmadmatrix"):])
    with open(json_path, "w") as f:
        json.dump(subprocesses, f)
    # written before compiling: _compile_subprocesses() reads the LIBTAG from it
    with open(marker_path, "w") as f:
        json.dump(marker, f)
    _driver._compile_subprocesses(rw_dir, _driver._read_devices(process_directory), fptype)
    return rw_dir


def run_parameter_reweighting(*args, regenerate=regenerate_process_in_model, **kwargs):
    """madtrex_driver.run_parameter_reweighting(), with "change model"
    support (see regenerate_process_in_model())."""
    return _driver.run_parameter_reweighting(*args, regenerate=regenerate, **kwargs)


def run_generic_reweighting(*args, regenerate=regenerate_process_in_model, **kwargs):
    """madtrex_driver.run_generic_reweighting(), with "change model"
    support (see regenerate_process_in_model())."""
    return _driver.run_generic_reweighting(*args, regenerate=regenerate, **kwargs)


def run_reweighting(card, events_path=None, card_kind=None, regenerate=regenerate_process_in_model,
                    **kwargs):
    """madtrex_driver.run_reweighting(): the reweight card through the
    parameter, PDF or generic runner its card_kind ("param", "pdf",
    "generic"; read from the card when None) needs, with "change model"
    support."""
    return _driver.run_reweighting(card, events_path=events_path, card_kind=card_kind,
                                   regenerate=regenerate, **kwargs)


def main(argv: list[str] | None = None) -> None:
    _driver.main(argv, regenerate=regenerate_process_in_model)


if __name__ == "__main__":
    main()
