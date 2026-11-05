"""Parsing for MadtRex's generic reweight card -- the card that lets one
card file mix parameter (matrix-element) and PDF reweighting, which
otherwise run as two entirely independent passes (see
param_reweight_card.py/pdf_reweight_card.py's own module docstrings, and
madtrex_driver.py's run_parameter_reweighting()/run_pdf_reweighting()) --
and the reweight-card vocabulary both MadtRex card readers share with the
classic reweighting module (madgraph/interface/reweight_interface.py).

Vocabulary: Cards/reweight_card.dat is read by either backend, so it uses
the classic module's spelling throughout -- "change <option> <value>" for
options of the reweighting itself, "set ..." only for physics parameters
(and, in a PDF section, the PDF set and scales). Where both backends have
the same option, it has the classic name and the classic default
("change helicity True"). The MadtRex spellings this replaces ("set
helicity_reweighting", "set FPTYPE", "set mode pdf", ...) are still
accepted, with a deprecation warning. The directives only the classic
module understands (see CLASSIC_ONLY_CHANGE_OPTIONS) are an error here
rather than silently ignored: they need "change backend classic".

The section grammar is deliberately minimal -- this module does not
understand either mode's own vocabulary, only:

- "change mode <LO|pdf>": switches which mode every following line belongs
  to, until the next "change mode" line or end of file. LO (the classic
  module's default mode) is parameter reweighting, pdf is PDF/scale
  reweighting. May appear more than once (switching back and forth), so
  launches of each mode can be interleaved in whatever order is most
  natural to write. Lines before the card's first "change mode" belong to
  LO, the default mode -- so a card without any "change mode" line is a
  plain parameter card. The former spelling "set mode <parameter|pdf>" is
  still accepted;
- everything else is opaque here: split_by_mode() just buckets each line's
  raw text by whichever mode is currently active, and hands the
  concatenated per-mode text (preserving each mode's own original line
  order) to that mode's real parser (param_reweight_card.parse_reweight_card()/
  pdf_reweight_card.parse_pdf_reweight_card()) completely unchanged -- so a
  "change mode LO" section's lines must themselves already be valid
  parameter-card syntax (change model/launch/set BLOCK ID VALUE/...), and
  likewise for "change mode pdf" (launch/set PDF_SET/set MUF/...).
  Concatenating same-mode chunks in file order preserves each mode's own
  preamble-vs-after-first-launch rules correctly, as long as (as is the
  natural way to write one) a mode's preamble-only directives appear in that
  mode's *first* section, before its own first "launch". An LO section
  without any "launch" (eg only "change helicity False" before a "change
  mode pdf") is dropped when another section has launches.

madtrex_driver.py's run_generic_reweighting() runs whichever mode(s)
split_by_mode() finds (both, if both are present) through their own real
routine, then coalesces the results back into a single output sample.
"""

import logging
import re

logger = logging.getLogger("madgraph7")

MODES = frozenset({"parameter", "pdf"})

# "change mode" values -> the section they select. LO is the classic
# module's name for parameter reweighting; "parameter" is the former
# "set mode parameter".
_MODE_NAMES = {"lo": "parameter", "parameter": "parameter", "pdf": "pdf"}

# The "change backend" values (see madgraph/various/reweight_backend.py).
BACKENDS = frozenset({"classic", "madtrex", "auto"})

# "change" options only the classic reweighting module understands --
# reweight_interface.REWEIGHT_CHANGE_OPTIONS/DENSITY_CHANGE_OPTIONS minus the
# shared ones (backend, model, process, helicity, mode). Kept here since this
# module ships in gridpacks without MadGraph; a unit test keeps the two in
# sync.
CLASSIC_ONLY_CHANGE_OPTIONS = frozenset({
    "keep_ordering", "use_eventid", "allow_missing_finalstate", "boost",
    "virtual_path", "tree_path", "output", "rwgt_dir", "systematics",
    "soft_threshold", "multicore", "identical_particle_in_prod_and_decay",
    "include_sudakov",
    # density-matrix reweighting (reweight_interface.DensityInterface)
    "helicity_direction", "boost_choice", "order_helicities",
    "symmetrise_initial_state", "axis_referential", "matrix_normalisation",
    "particle_in_density_matrix",
})
# "change mode" values only the classic module understands.
CLASSIC_ONLY_MODES = frozenset({"nlo", "nlo_tree", "lo+nlo"})

_MODE_RE = re.compile(r"^\s*(?P<verb>change|set)\s+mode\s+(?P<mode>\S+)\s*$", re.I)
_LAUNCH_RE = re.compile(r"^\s*launch\b", re.I)


# A "!" comment: from a "!" at the start of the line or after whitespace (the
# reweight cards' examples use both "#" and "!" for trailing comments).
_BANG_COMMENT_RE = re.compile(r"(?:^|\s)!.*$")


def strip_comment(line: str) -> str:
    """line without its comment: what follows a "#", or a "!" at the start
    of the line or after whitespace."""
    return _BANG_COMMENT_RE.sub("", line.split("#", 1)[0])


def deprecated(old: str, new: str) -> None:
    """Warns that the reweight-card spelling old is replaced by new."""
    logger.warning("reweight card: '%s' is deprecated, use '%s' instead", old, new)


def classic_only_reason(line: str) -> str | None:
    """Why line (comment already stripped) needs the classic reweighting
    module, or None when MadtRex may understand it."""
    words = line.split()
    if len(words) < 2 or words[0].lower() != "change":
        if words and words[0].lower() not in ("set", "change", "launch") and (
            "/" in words[0] or words[0].endswith(".dat")
        ):
            return "a param_card given by path"
        return None
    name = words[1].lower()
    if name in CLASSIC_ONLY_CHANGE_OPTIONS:
        return f"'change {name}' is a classic reweighting option"
    if name == "mode" and len(words) > 2 and words[2].lower() in CLASSIC_ONLY_MODES:
        return "NLO reweighting"
    if name == "process" and words[-1] == "--add":
        return "several processes for one model"
    return None


def check_not_classic_only(line: str, raw_line: str) -> None:
    """Raises if line needs the classic reweighting module."""
    reason = classic_only_reason(line)
    if reason is not None:
        raise ValueError(
            f"reweight card line {raw_line.strip()!r} needs the classic reweighting "
            f"module ({reason}), not MadtRex: set 'change backend classic' (or remove "
            f"the line)"
        )


def check_backend_line(words: list[str], raw_line: str) -> None:
    """Checks a "change backend <value>" line read by MadtRex."""
    value = words[2].lower() if len(words) > 2 else ""
    if value not in BACKENDS:
        raise ValueError(
            f"reweight card line {raw_line.strip()!r}: unknown backend {value!r}; "
            f"expected one of {sorted(BACKENDS)}"
        )
    if value == "classic":
        raise ValueError(
            f"reweight card line {raw_line.strip()!r} asks for the classic reweighting "
            f"module, not MadtRex"
        )


CARD_KINDS = ("param", "pdf", "generic")
# "set" names and "change" options of a PDF reweighting card
_PDF_SET_NAMES = frozenset({
    "pdf_set", "beam1_pdf_set", "beam2_pdf_set", "muf", "mur", "muf_factor", "mur_factor",
})
_PDF_CHANGE_OPTIONS = frozenset({"alpha_s_power", "scale_variation_output"})


def card_kind(text: str) -> str:
    """Which MadtRex runner reads the card text: "generic" when it has a
    "change mode" (or former "set mode") line, "pdf" when it names a PDF set,
    scale or PDF-reweighting option, else "param". Same rule as
    madgraph/various/reweight_backend.classify_card()."""
    has_pdf = False
    for raw_line in text.splitlines():
        words = strip_comment(raw_line).split()
        if len(words) < 2:
            continue
        keyword, name = words[0].lower(), words[1].lower()
        if keyword in ("change", "set") and name == "mode":
            return "generic"
        if (keyword == "set" and len(words) == 3 and name in _PDF_SET_NAMES | _PDF_CHANGE_OPTIONS) \
                or (keyword == "change" and name in _PDF_CHANGE_OPTIONS):
            has_pdf = True
    return "pdf" if has_pdf else "param"


def split_by_mode(text: str) -> dict[str, str]:
    """Splits text into {mode: sub_card_text} ("parameter" for LO, "pdf"),
    one entry per mode actually used (in first-appearance order), each
    sub_card_text being every line that belonged to that mode, concatenated
    in original file order and ready to hand to that mode's own parser
    as-is. Lines before the first "change mode" are LO.

    Raises if a "change mode" line names anything other than LO/pdf (or the
    former "set mode parameter|pdf")."""
    chunks: dict[str, list[str]] = {"parameter": []}
    has_launch = {"parameter": False}
    current_mode = "parameter"

    for raw_line in text.splitlines():
        line = strip_comment(raw_line)

        mode_match = _MODE_RE.match(line) if line.strip() else None
        if mode_match:
            value = mode_match.group("mode").strip()
            mode = _MODE_NAMES.get(value.lower())
            if mode is None:
                if value.lower() in CLASSIC_ONLY_MODES:
                    check_not_classic_only(line, raw_line)
                raise ValueError(
                    f"generic reweight card: unknown reweighting mode {value!r} in "
                    f"{raw_line!r}; expected 'change mode LO' or 'change mode pdf'"
                )
            if mode_match.group("verb").lower() == "set":
                deprecated(f"set mode {value}", "change mode " + ("LO" if mode == "parameter" else "pdf"))
            current_mode = mode
            chunks.setdefault(current_mode, [])
            has_launch.setdefault(current_mode, False)
            continue

        if _LAUNCH_RE.match(line):
            has_launch[current_mode] = True
        chunks[current_mode].append(raw_line)

    if not has_launch["parameter"] and len(chunks) > 1:
        del chunks["parameter"]
    return {mode: "\n".join(lines) for mode, lines in chunks.items()}
