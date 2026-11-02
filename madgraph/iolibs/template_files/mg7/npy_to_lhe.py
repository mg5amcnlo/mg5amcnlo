"""Convert an mg7 numpy event file (``events.npy``) into an LHE file.

The mg7 output writes its events as npy by default. Anything that needs an LHE
file -- a shower, MadSpin, reweighting, an analysis -- normally switches the run
to the LHE format before the generation (``force_lhe_output_if_needed``). This
module covers the case where that request comes *after* the run: it rebuilds
the LHE file from what the npy run left behind.

* ``output_format = "lhe_npy"``: the file already holds complete LHE events
  (pdg ids, status, mothers, colour, helicity); they are only reformatted.
* ``output_format = "compact_npy"``: the file holds the momenta plus the
  subprocess/diagram/colour/flavour/helicity indices the generation picked.
  Those are completed exactly as the direct LHE output does it, with the
  :class:`madspace.LHECompleter` the run saved next to the events
  (``lhe_completer.json``). The completion draws random numbers (colour flow
  among equivalent ones, resonance assignment), so the result is statistically
  but not byte-identical to an ``output_format = "lhe"`` run with the same seed.

In both cases the LHE header and ``<init>`` block (with the systematics
``<initrwgt>``) are taken verbatim from the ``header.lhe`` file the run writes
next to its npy events; the ``rwgt_<id>`` columns become the per-event
``<rwgt>`` block: the systematics' numbered ones, then the named ones MadtRex
reweighting adds (``rwgt_<name>``, listed in header.lhe's ``<initrwgt>``). The ``<mgrwt>`` reweighting-input block (``[systematics]
write_inputs``) is not reconstructed.

Usage::

    bin/npy_to_lhe [RUN_NAME | RUN_DIR | path/to/events.npy] [-o OUTPUT] [--seed N]
"""

import argparse
import logging
import os
import re
import sys

logger = logging.getLogger("madevent")

COMPLETER_FILE = "lhe_completer.json"
HEADER_FILE = "header.lhe"
EVENT_FILE = "events.npy"

# events converted per LHEFileWriter.write_string call
_CHUNK = 10000


def _madspace():
    from madgraph.iolibs.template_files.mg7.bootstrap import ensure_madspace
    ensure_madspace(interactive=False)
    import madspace
    return madspace


def _read_header(run_path):
    """The header.lhe text without its closing tag, ready to be followed by
    the events."""
    with open(os.path.join(run_path, HEADER_FILE)) as f:
        text = f.read()
    end = text.rfind("</LesHouchesEvents>")
    return text if end < 0 else text[:end]


def _seed(header, seed):
    if seed is not None:
        return int(seed)
    match = re.search(r"<MG7Seed>\s*(-?\d+)\s*</MG7Seed>", header)
    return int(match.group(1)) if match else 0


# ----------------------------------------------------------------------------
# conversion
# ----------------------------------------------------------------------------
def resolve_event_file(path, me_dir=None) -> str:
    """``path`` may be an npy file, a run directory or a run name (looked up
    under ``<me_dir>/Events``)."""
    candidates = [path, os.path.join(path, EVENT_FILE)]
    if me_dir is not None:
        candidates.append(os.path.join(me_dir, "Events", path, EVENT_FILE))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError("no %s found for '%s'" % (EVENT_FILE, path))


def can_convert(run_path) -> bool:
    """True when ``run_path`` holds npy events this module can turn into LHE."""
    return (os.path.isfile(os.path.join(run_path, EVENT_FILE))
            and os.path.isfile(os.path.join(run_path, HEADER_FILE)))


def _particle_count(names):
    count = 0
    while "part%d_energy" % (count + 1) in names:
        count += 1
    return count


def _compact_events(ms, events, completer, columns, rng):
    names = events.dtype.names
    nparticles = _particle_count(names)
    momenta = [tuple(events["part%d_%s" % (i + 1, c)]
                     for c in ("energy", "px", "py", "pz"))
               for i in range(nparticles)]
    rwgt = [events[c] for c in columns]
    for n in range(len(events)):
        particles = []
        for energy, px, py, pz in momenta:
            if energy[n] == 0.:
                break
            particles.append(ms.LHEParticle(px=px[n], py=py[n], pz=pz[n],
                                            energy=energy[n]))
        event = ms.LHEEvent(weight=events["weight"][n],
                            scale=events["ren_scale"][n],
                            alpha_qcd=events["alpha_qcd"][n],
                            particles=particles)
        completer.complete_event_data(
            event, int(events["subprocess_index"][n]),
            int(events["diagram_index"][n]), int(events["color_index"][n]),
            int(events["flavor_index"][n]), int(events["helicity_index"][n]),
            rng)
        yield event, [float(w[n]) for w in rwgt]


_LHE_PARTICLE_FIELDS = ("pdg_id", "status_code", "mother1", "mother2", "color",
                        "anti_color", "px", "py", "pz", "energy", "mass",
                        "lifetime", "spin")


def _lhe_npy_events(ms, events, columns):
    names = events.dtype.names
    count = 0
    while "part%d_pdg_id" % (count + 1) in names:
        count += 1
    fields = [[events["part%d_%s" % (i + 1, f)] for f in _LHE_PARTICLE_FIELDS]
              for i in range(count)]
    rwgt = [events[c] for c in columns]
    for n in range(len(events)):
        particles = []
        for values in fields:
            if values[0][n] == 0:
                break
            particles.append(ms.LHEParticle(
                *[v[n].item() for v in values]))
        event = ms.LHEEvent(int(events["process_id"][n]),
                            events["weight"][n], events["scale"][n],
                            events["alpha_qed"][n], events["alpha_qcd"][n],
                            particles)
        yield event, [float(w[n]) for w in rwgt]


def _add_named_weights(text, names, values):
    """The formatted LHE event text with the named weights (ids madspace's
    integer-id ``rwgt`` cannot carry) appended to its ``<rwgt>`` block, which
    is created when the event has none."""
    if not names:
        return text
    extra = "".join("<wgt id='%s'> %+.7e </wgt>\n" % (name, value)
                    for name, value in zip(names, values))
    if "</rwgt>" in text:
        return text.replace("</rwgt>", extra + "</rwgt>", 1)
    return text.replace("</event>", "<rwgt>\n" + extra + "</rwgt>\n</event>", 1)


def convert(path, output=None, seed=None, compress=True, me_dir=None) -> str:
    """Write the LHE file for the npy events at ``path`` (npy file, run
    directory or run name) and return its path. By default it is written as
    ``events.lhe.gz`` next to the npy file."""
    import numpy as np
    from madgraph.various import misc

    ms = _madspace()
    npy_path = resolve_event_file(path, me_dir)
    run_path = os.path.dirname(npy_path)
    # A standalone conversion starts in a fresh process, so it has not seen
    # the gzip configuration installed by the event-generation launcher.
    card_root = me_dir or os.path.dirname(os.path.dirname(run_path))
    card_path = os.path.join(card_root, "Cards", "run_card.toml")
    if os.path.isfile(card_path):
        from madgraph.various.banner import RunCardMG7
        misc.configure_gzip(RunCardMG7(card_path).gzip_configuration())
    if not os.path.isfile(os.path.join(run_path, HEADER_FILE)):
        raise FileNotFoundError(
            "%s is missing: the run predates the npy->LHE conversion support "
            "and cannot be converted." % os.path.join(run_path, HEADER_FILE))
    header = _read_header(run_path)

    events = np.load(npy_path, mmap_mode="r", max_header_size=1 << 30)
    names = events.dtype.names
    columns = [n for n in names if re.fullmatch(r"rwgt_\d+", n)]
    ids = [int(c.split("_", 1)[1]) for c in columns]
    # MadtRex reweighting's named weights (rwgt_<name>)
    named = [n for n in names if n.startswith("rwgt_") and n not in columns]
    named_ids = [n[len("rwgt_"):] for n in named]
    if "diagram_index" in names:
        completer_path = os.path.join(run_path, COMPLETER_FILE)
        if not os.path.isfile(completer_path):
            raise FileNotFoundError("%s is missing" % completer_path)
        completer = ms.LHECompleter.load(completer_path)
        rng = ms.MixMaxRandom(_seed(header, seed))
        stream = lambda chunk: _compact_events(ms, chunk, completer, columns, rng)
    elif "part1_pdg_id" in names:
        stream = lambda chunk: _lhe_npy_events(ms, chunk, columns)
    else:
        raise ValueError("%s is not an mg7 event file" % npy_path)

    if output is None:
        output = os.path.join(run_path, "events.lhe")
    if output.endswith(".gz"):
        output, compress = output[:-3], True
    logger.info("converting %s (%d events) to LHE", npy_path, len(events))
    with open(output, "w") as out:
        out.write(header)
        for start in range(0, len(events), _CHUNK):
            chunk = np.asarray(events[start:start + _CHUNK])
            text = []
            named_values = [chunk[n] for n in named]
            for i, (event, weights) in enumerate(stream(chunk)):
                if ids:
                    event.rwgt_ids = ids
                    event.rwgt = weights
                text.append(_add_named_weights(event.format(), named_ids,
                                               [float(v[i]) for v in named_values]))
            out.write("".join(text))
        out.write("</LesHouchesEvents>\n")
    if compress:
        misc.gzip(output)
        output += ".gz"
    logger.info("LHE events written to %s", output)
    return output


def main(argv=None, me_dir=None) -> None:
    """Command-line entry point; run names are looked up under
    ``<me_dir>/Events`` (default: the working directory)."""
    parser = argparse.ArgumentParser(
        description="Convert an mg7 events.npy file into an LHE file.")
    parser.add_argument("path", nargs="?", default=None,
                        help="run name, run directory or events.npy file "
                             "(default: the most recent run with npy events "
                             "and no LHE file)")
    parser.add_argument("-o", "--output", default=None,
                        help="output file (default: events.lhe.gz next to the "
                             "npy file)")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed of the LHE completion (default: the run seed)")
    parser.add_argument("--no-gzip", action="store_false", dest="compress",
                        help="do not compress the output")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    me_dir = me_dir or os.getcwd()
    path = args.path
    if path is None:
        events_dir = os.path.join(me_dir, "Events")
        runs = [os.path.join(events_dir, d) for d in
                (os.listdir(events_dir) if os.path.isdir(events_dir) else [])]
        runs = [d for d in runs if can_convert(d) and not any(
            os.path.exists(os.path.join(d, n))
            for n in ("events.lhe", "events.lhe.gz"))]
        if not runs:
            sys.exit("no run with unconverted npy events in %s" % events_dir)
        path = max(runs, key=lambda d: os.path.getmtime(
            os.path.join(d, EVENT_FILE)))
    try:
        convert(path, args.output, args.seed, args.compress, me_dir)
    except (FileNotFoundError, ValueError) as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
