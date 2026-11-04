"""The model of a process directory and its auto widths. Does not import madspace."""

import logging
import os
import re
import subprocess

from madgraph.models.check_param_card import group_auto_widths
from madgraph.various import misc
from madgraph.mg7.bootstrap import MG_ROOT as _MG_ROOT

logger = logging.getLogger("madgraph7")


_model_hash_warned = set()
_model_cache = {}


def read_stored_model(me_dir=None) -> "str | None":
    """The model this process was generated with, as recorded at output time in
    ``SubProcesses/model.txt``: a UFO path carrying its restriction (e.g.
    ``.../models/sm-no_b_mass``) or a model name -- in either case something
    ``import model`` understands. Returns None when the process was written
    without that information.

    Warns (once per directory) when the model on disk no longer matches the
    hash stored on the second line, i.e. it changed since the output was made.
    """
    base = me_dir or os.getcwd()
    model_file = os.path.join(base, "SubProcesses", "model.txt")
    if not os.path.exists(model_file):
        return None
    with open(model_file) as f:
        lines = f.read().splitlines()
    model = lines[0].strip() if lines else ""
    stored_hash = lines[1].strip() if len(lines) > 1 else ""
    if not model:
        logger.warning("SubProcesses/model.txt is empty; the model of this "
                       "process is unknown.")
        return None

    # verify the model on disk still matches the one used at output time. The
    # reference may carry a restriction suffix ('sm-no_b_mass'); the hash was
    # taken on the UFO directory itself ('sm').
    model_dir = model if os.path.isdir(model) else model.rsplit('-', 1)[0]
    if stored_hash and os.path.isdir(model_dir) \
            and model_file not in _model_hash_warned:
        current_hash = misc.hash_model_files(model_dir)
        if current_hash and current_hash != stored_hash:
            _model_hash_warned.add(model_file)
            logger.warning(
                "The model at %s has changed since this process was generated "
                "(hash mismatch); anything computed from it now (the 'auto' "
                "widths, the dependent masses/widths of the param_card) may be "
                "inconsistent with the matrix element.", model_dir)
    return model


def _proc_characteristic(me_dir=None) -> dict:
    """SubProcesses/proc_characteristics as a plain dict (empty when the file is
    missing or unreadable). Plain, because ProcCharacteristic is a ConfigFile,
    whose get() is the parameter accessor and takes no default."""
    from madgraph.various import banner as _banner_mod
    path = os.path.join(me_dir or os.getcwd(), 'SubProcesses',
                        'proc_characteristics')
    if os.path.exists(path):
        try:
            return dict(_banner_mod.ProcCharacteristic(path))
        except Exception as error:
            logger.debug("could not read %s: %s", path, error)
    return {}


def load_process_model(me_dir=None):
    """Import the UFO model this process was generated with, as
    :func:`madgraph.models.import_ufo.import_model` builds it -- the object every
    model-level consumer of the param_card expects (see
    :meth:`MG7Cmd.get_model`). None when the process recorded no model, or when
    that model no longer imports.

    Cached per (reference, complex-mass-scheme): importing a model is slow and
    the card question asks for it again at every 'update dependent'."""
    ref = read_stored_model(me_dir)
    if not ref:
        return None
    cms = _proc_characteristic(me_dir).get('complex_mass_scheme', False)
    key = (ref, bool(cms))
    if key not in _model_cache:
        import madgraph.models.import_ufo as import_ufo
        try:
            with misc.MuteLogger(['madgraph.model'], [50]):
                _model_cache[key] = import_ufo.import_model(
                    ref, complex_mass_scheme=bool(cms))
        except Exception as error:
            # 'update dependent' calls this under a SIGALRM whose handler
            # raises a TimeOutError of its own (a class defined inside the
            # caller, so it can only be recognised by name). Let it through:
            # the caller then says the model took too long to load and how to
            # force it, instead of reporting a model that does not import.
            if type(error).__name__ == 'TimeOutError':
                raise
            logger.debug("could not import the model %s: %s", ref, error)
            _model_cache[key] = None
    return _model_cache[key]


def compute_auto_widths(param_card_path=os.path.join("Cards", "param_card.dat")) -> None:
    """Fill any width set to ``auto`` in the param_card, using madgraph and the
    model stored at output time (``SubProcesses/model.txt``), and write the
    result back into the card. A no-op when the card has no ``auto`` width.

    Called before every generation, so a single run and each scan point (whose
    param_card is rewritten just before ``run_single``) both get model-computed
    widths."""
    try:
        with open(param_card_path) as f:
            text = f.read()
    except OSError:
        return
    # "DECAY <pdg> auto", with the options of compute_widths that may follow
    # ("auto@NLO", "auto --nlo --body_decay=3"), as in
    # common_run_interface.static_check_param_card: one compute_widths per set of
    # options
    groups = group_auto_widths(text)
    if not groups:
        return
    pdgs = [pdg for group in groups.values() for pdg in group]

    model = read_stored_model()
    if not model:
        logger.warning(
            "The param_card requests 'auto' width(s) for %s but the model of "
            "this process is unknown; leaving them as-is.", " ".join(pdgs))
        return

    mg5 = str(_MG_ROOT / "bin" / "madgraph")
    if not os.path.exists(mg5):
        logger.warning("Cannot find madgraph at %s; 'auto' widths not computed.", mg5)
        return

    import tempfile
    cmds = "import model %s\n" % model
    for options, group in groups.items():
        cmds += "compute_widths %s --path=%s %s\n" % (
            " ".join(group), os.path.abspath(param_card_path), " ".join(options))
    with tempfile.NamedTemporaryFile("w", suffix=".mg5", delete=False) as fh:
        fh.write(cmds)
        cmdfile = fh.name
    try:
        logger.info("Computing 'auto' width(s) for %s ...", " ".join(pdgs))
        proc = subprocess.run([mg5, cmdfile])
        if proc.returncode != 0:
            logger.warning(
                "compute_widths returned a non-zero exit code; the param_card "
                "may still contain 'auto' entries.")
    finally:
        try:
            os.remove(cmdfile)
        except OSError:
            pass
